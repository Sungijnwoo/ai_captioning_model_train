import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Flickr8kCaptionDataset import Flickr8kCaptionDataset
from domain.dto.AlignConfig import AlignConfig
from model.ClipToLlmProjector import ClipToLlmProjector
from util.Logger import Logger
from util.Util import Util


class AlignTrainer:
    def __init__(self, align_config: AlignConfig):
        self.logger = Logger.get_logger(__name__)
        self.config = align_config
        self.accelerator = Accelerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 세팅
        self.clip_model, self.clip_preprocess_train, self.clip_preprocess_val = Util.load_clip_model(
            self.config.clip_model_path, self.config.clip_model_id
        )
        self.llm_model, self.llm_tokenizer = Util.load_llm_model(self.config.llm_model_id, is_freeze=True)

        # Dim 구하기
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = int(self.clip_model.encode_image(dummy).shape[-1])
        llm_dim = int(self.llm_model.config.hidden_size)
        self.clip_model.to(self.device)
        self.llm_model.to(self.device)
        self.projector = ClipToLlmProjector(clip_dim, llm_dim, self.config.project_multi_rate).to(self.device)
        self.logger.info(f"Projector 모델 세팅 완료 {clip_dim} -> {llm_dim}")

        # 데이터 세팅
        train_dataset = Flickr8kCaptionDataset(
            align_config.image_path,
            align_config.text_path,
            45,
            split_path=align_config.train_split_path,
        )
        val_dataset = Flickr8kCaptionDataset(
            align_config.image_path,
            align_config.text_path,
            45,
            split_path=align_config.val_split_path,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=align_config.batch_size,
            shuffle=True,
            num_workers=align_config.num_worker,
            collate_fn=Flickr8kCaptionDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=align_config.batch_size,
            shuffle=False,
            num_workers=align_config.num_worker,
            collate_fn=Flickr8kCaptionDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.logger.info("데이터 세팅 완료")

        self.optim = torch.optim.AdamW(self.projector.parameters(), lr=align_config.align_lr, weight_decay=0.01)
        self.projector, self.optim, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.projector, self.optim, self.train_loader, self.val_loader
        )
        self.logger.info("학습 세팅 완료")

    def __mean_pool(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def __encode_text_pooled(self, captions: list[str]) -> torch.Tensor:
        tok = self.llm_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.device)
        attn = tok.attention_mask.to(self.device)
        emb_layer = self.llm_model.get_input_embeddings()
        hidden = emb_layer(input_ids)
        return self.__mean_pool(hidden, attn)

    def __contrastive_clip_llm_loss(
        self,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        *,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        z_img = F.normalize(z_img, dim=-1)
        z_txt = F.normalize(z_txt, dim=-1)
        logits = (z_img @ z_txt.T) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.T, targets)
        return (loss_i + loss_t) / 2

    def __compute_loss(self, batch: dict, preprocess: callable) -> torch.Tensor:
        pixels = torch.stack([preprocess(im) for im in batch["images"]]).to(self.device)
        with torch.no_grad():
            img_raw = self.clip_model.encode_image(pixels)

        z_img = self.projector(img_raw)
        z_txt = self.__encode_text_pooled(batch["captions"])
        return self.__contrastive_clip_llm_loss(z_img, z_txt, temperature=self.config.align_temperature)

    @torch.no_grad()
    def __validate(self) -> float:
        self.projector.eval()
        losses: list[float] = []

        for batch in self.val_loader:
            loss = self.__compute_loss(batch, self.clip_preprocess_val)
            loss_f = float(self.accelerator.gather(loss.detach().unsqueeze(0)).mean())
            losses.append(loss_f)

        self.projector.train()
        return sum(losses) / max(1, len(losses))

    def train(self):
        formatted = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        out_dir = Path(self.config.output_dir) / formatted
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.projector.train()
            global_step = 0
            losses: list[float] = []
            val_losses: list[float] = []
            best_val_loss = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0
            stopped_early = False

            for epoch in range(self.config.align_epochs):
                pbar = tqdm(
                    self.train_loader,
                    desc=f"align epoch {epoch + 1}/{self.config.align_epochs}",
                    disable=not self.accelerator.is_local_main_process,
                )
                for batch in pbar:
                    loss = self.__compute_loss(batch, self.clip_preprocess_train)
                    self.accelerator.backward(loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    loss_f = float(self.accelerator.gather(loss.detach().unsqueeze(0)).mean())
                    losses.append(loss_f)
                    global_step += 1
                    pbar.set_postfix(loss=f"{loss_f:.03f}")

                val_loss = self.__validate()
                val_losses.append(val_loss)
                self.logger.info(f"align epoch {epoch + 1} validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0

                    if self.accelerator.is_main_process:
                        self.config.save(out_dir / "align_config.yaml")
                        torch.save(self.accelerator.unwrap_model(self.projector).state_dict(), out_dir / "projector.pt")
                        self.logger.info(f"align best 모델 저장 완료: epoch={best_epoch}, val_loss={best_val_loss:.4f}")
                else:
                    epochs_without_improvement += 1

                if (
                    self.config.early_stop_patience > 0
                    and epochs_without_improvement >= self.config.early_stop_patience
                ):
                    stopped_early = True
                    self.logger.info(
                        f"align early stopping at epoch {epoch + 1} (best_epoch={best_epoch}, patience={self.config.early_stop_patience})"
                    )
                    break

            if self.accelerator.is_main_process:
                torch.save(self.accelerator.unwrap_model(self.projector).state_dict(), out_dir / "last_projector.pt")
                with (out_dir / "align_log.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "epochs_completed": len(val_losses),
                            "steps": global_step,
                            "mean_loss": sum(losses) / max(1, len(losses)),
                            "mean_val_loss": sum(val_losses) / max(1, len(val_losses)),
                            "last_val_loss": val_losses[-1] if val_losses else None,
                            "best_val_loss": best_val_loss if val_losses else None,
                            "best_epoch": best_epoch if val_losses else None,
                            "stopped_early": stopped_early,
                        },
                        f,
                        indent=2,
                    )
                self.logger.info(f"{out_dir}에 align 모델 저장 완료")

            self.accelerator.wait_for_everyone()
        except BaseException:
            if self.accelerator.is_main_process and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
                self.logger.info(f"학습 중단으로 미완료 출력 폴더 삭제: {out_dir}")
            raise
