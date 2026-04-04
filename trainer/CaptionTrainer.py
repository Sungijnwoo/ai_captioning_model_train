import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Flickr8kCaptionDataset import Flickr8kCaptionDataset
from domain.dto.CaptionConfig import CaptionConfig
from model.ClipToLlmProjector import ClipToLlmProjector
from util.Logger import Logger
from util.Util import Util


class CaptionTrainer:
    def __init__(self, caption_config: CaptionConfig):
        self.logger = Logger.get_logger(__name__)
        self.caption_config = caption_config
        self.accelerator = Accelerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 세팅
        self.clip_model, self.clip_preprocess_train, self.clip_preprocess_val = Util.load_clip_model(
            self.caption_config.clip_model_path, self.caption_config.clip_model_id
        )
        self.llm_model, self.llm_tokenizer = Util.load_llm_model(self.caption_config.llm_model_id)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = int(self.clip_model.encode_image(dummy).shape[-1])
        llm_dim = int(self.llm_model.config.hidden_size)

        self.clip_model.to(self.device)
        self.llm_model.to(self.device)
        self.projector = ClipToLlmProjector(clip_dim, llm_dim, self.caption_config.project_multi_rate).to(self.device)
        projector_path = Path(self.caption_config.project_model_dir) / "projector.pt"
        state_dict = torch.load(projector_path, map_location="cpu")
        self.projector.load_state_dict(state_dict)
        self.logger.info(f"학습된 projector 로드 완료: {projector_path}")

        dataset = Flickr8kCaptionDataset(caption_config.image_path, caption_config.text_path, 45)
        self.train_loader = DataLoader(
            dataset,
            batch_size=caption_config.batch_size,
            shuffle=True,
            num_workers=caption_config.num_worker,
            collate_fn=Flickr8kCaptionDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.logger.info("Caption 데이터 세팅 완료")

        trainable_params = list(self.projector.parameters()) + list(self.llm_model.parameters())
        self.optim = torch.optim.AdamW(trainable_params, lr=caption_config.caption_lr, weight_decay=0.01)
        self.projector, self.llm_model, self.optim, self.train_loader = self.accelerator.prepare(
            self.projector, self.llm_model, self.optim, self.train_loader
        )
        self.logger.info("학습 세팅 완료")

    def __tokenize_captions(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.llm_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.caption_config.max_text_length,
            return_tensors="pt",
        )
        return tok.input_ids.to(self.device), tok.attention_mask.to(self.device)

    def __build_inputs(
        self, image_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        image_token = image_embeds.unsqueeze(1)
        inputs_embeds = torch.cat([image_token, text_embeds], dim=1)

        image_attention = torch.ones(
            (attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([image_attention, attention_mask], dim=1)

        labels = input_ids.masked_fill(attention_mask == 0, -100)
        image_labels = torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype)
        full_labels = torch.cat([image_labels, labels], dim=1)

        return inputs_embeds, full_attention_mask, full_labels

    def train(self):
        out_dir = Path(self.caption_config.output_dir) / "caption"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.projector.train()
        self.llm_model.train()
        global_step = 0
        losses: list[float] = []

        for epoch in range(self.caption_config.caption_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"caption epoch {epoch + 1}/{self.caption_config.caption_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in pbar:
                pixels = torch.stack([self.clip_preprocess_train(im) for im in batch["images"]]).to(self.device)

                with torch.no_grad():
                    img_raw = self.clip_model.encode_image(pixels)

                image_embeds = self.projector(img_raw)
                input_ids, attention_mask = self.__tokenize_captions(batch["captions"])
                inputs_embeds, full_attention_mask, labels = self.__build_inputs(
                    image_embeds, input_ids, attention_mask
                )

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                self.accelerator.backward(loss)
                self.optim.step()
                self.optim.zero_grad()

                loss_f = float(self.accelerator.gather(loss.detach().unsqueeze(0)).mean())
                losses.append(loss_f)
                global_step += 1
                pbar.set_postfix(loss=f"{loss_f:.03f}")

        if self.accelerator.is_main_process:
            projector_to_save = self.accelerator.unwrap_model(self.projector)
            llm_to_save = self.accelerator.unwrap_model(self.llm_model)
            self.caption_config.save(out_dir / "caption_config.yaml")
            torch.save(projector_to_save.state_dict(), out_dir / "projector.pt")
            llm_to_save.save_pretrained(out_dir / "llm")
            self.llm_tokenizer.save_pretrained(out_dir / "llm")

            with (out_dir / "caption_log.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"steps": global_step, "mean_loss": sum(losses) / max(1, len(losses))},
                    f,
                    indent=2,
                )
            self.logger.info(f"{out_dir}에 모델 저장 완료")

        self.accelerator.wait_for_everyone()
