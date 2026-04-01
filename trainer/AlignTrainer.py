import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Flickr8kCaptionDataset import Flickr8kCaptionDataset
from domain.dto.Config import Config
from model.ClipToLlmProjector import ClipToLlmProjector
from util.Logger import Logger
from util.Util import Util


class AlignTrainer:
    def __init__(self, config: Config):
        self.logger = Logger.get_logger(__name__)
        self.config = config
        self.accelerator = Accelerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 세팅
        self.clip_model, self.clip_preprocess_train, self.clip_preprocess_val = Util.load_clip_model(self.config)
        self.llm_model, self.llm_tokenizer = Util.load_llm_model(self.config)

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
        dataset = Flickr8kCaptionDataset(config.image_path, config.text_path, 45)
        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_worker,
            collate_fn=Flickr8kCaptionDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.logger.info("데이터 세팅 완료")

        self.optim = torch.optim.AdamW(self.projector.parameters(), lr=config.align_lr, weight_decay=0.01)
        self.projector, self.optim, self.train_loader = self.accelerator.prepare(
            self.projector, self.optim, self.train_loader
        )

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

    def train(self):
        out_dir = Path(self.config.output_dir) / "align"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.projector.train()
        global_step = 0
        losses: list[float] = []

        for epoch in range(self.config.align_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"align epoch {epoch + 1}/{self.config.align_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in pbar:
                pixels = torch.stack([self.clip_preprocess_train(im) for im in batch["images"]]).to(self.device)
                with torch.no_grad():
                    img_raw = self.clip_model.encode_image(pixels)

                z_img = self.projector(img_raw)
                z_txt = self.__encode_text_pooled(batch["captions"])

                loss = self.__contrastive_clip_llm_loss(z_img, z_txt, temperature=self.config.align_temperature)
                self.accelerator.backward(loss)
                self.optim.step()
                self.optim.zero_grad()

                loss_f = float(self.accelerator.gather(loss.detach().unsqueeze(0)).mean())
                losses.append(loss_f)
                global_step += 1
                pbar.set_postfix(loss=loss_f)

        if self.accelerator.is_main_process:
            self.logger.info(f"steps: {global_step} mean_loss: {sum(losses / max(1, len(losses)))}")
            torch.save(self.accelerator.unwrap_model(self.projector).state_dict(), out_dir / "projector.pt")
            with (out_dir / "align_log.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"steps": global_step, "mean_loss_last_100": sum(losses[-100:]) / max(1, len(losses[-100:]))},
                    f,
                    indent=2,
                )

        self.accelerator.wait_for_everyone()
