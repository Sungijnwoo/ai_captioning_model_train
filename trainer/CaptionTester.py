import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.Flickr8kCaptionDataset import Flickr8kCaptionDataset
from domain.dto.CaptionConfig import CaptionConfig
from domain.dto.TestConfig import TestConfig
from model.ClipToLlmProjector import ClipToLlmProjector
from util.Logger import Logger
from util.Util import Util


class CaptionTester:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        llm_name: str = "llm",
        projector_name: str = "projector.pt",
    ):
        self.logger = Logger.get_logger(__name__)
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.llm_name = llm_name
        self.projector_name = projector_name

        config_path = self.model_dir / "caption_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"caption config not found: {config_path}")

        self.config = CaptionConfig.load_config(str(config_path))
        self.clip_model, _, self.clip_preprocess = Util.load_clip_model(
            self.config.clip_model_path, self.config.clip_model_id
        )
        self.llm_model, self.llm_tokenizer = self.__load_llm()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = int(self.clip_model.encode_image(dummy).shape[-1])
        llm_dim = int(self.llm_model.config.hidden_size)

        self.projector = ClipToLlmProjector(clip_dim, llm_dim, self.config.project_multi_rate).to(self.device)
        projector_path = self.model_dir / self.projector_name
        projector_state = torch.load(projector_path, map_location="cpu")
        self.projector.load_state_dict(projector_state)

        self.clip_model.to(self.device).eval()
        self.llm_model.to(self.device).eval()
        self.projector.eval()
        self.logger.info(f"caption 모델 로드 완료: {self.model_dir}")

    def __load_llm(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        llm_dir = self.model_dir / self.llm_name
        if llm_dir.exists():
            llm_model = AutoModelForCausalLM.from_pretrained(llm_dir)
            llm_tokenizer = AutoTokenizer.from_pretrained(llm_dir)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            return llm_model, llm_tokenizer

        return Util.load_llm_model(self.config.llm_model_id)

    @staticmethod
    def __load_references(token_path: str, split_path: Optional[str]) -> Dict[str, List[str]]:
        allowed_images = Flickr8kCaptionDataset.load_image_names(split_path)
        return Flickr8kCaptionDataset.load_grouped_captions(token_path, allowed_images)

    @staticmethod
    def __tokenize_text(text: str) -> List[str]:
        return [tok for tok in text.lower().strip().split() if tok]

    @staticmethod
    def __ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def __corpus_bleu(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]],
        weights: List[float],
    ) -> float:
        precisions: List[float] = []

        for n, weight in enumerate(weights, start=1):
            if weight == 0:
                precisions.append(1.0)
                continue

            clipped_total = 0
            hypothesis_total = 0
            for refs, hyp in zip(references, hypotheses):
                hyp_ngrams = self.__ngrams(hyp, n)
                hypothesis_total += sum(hyp_ngrams.values())
                if not hyp_ngrams:
                    continue

                max_ref_counts: Counter[Tuple[str, ...]] = Counter()
                for ref in refs:
                    ref_ngrams = self.__ngrams(ref, n)
                    for ngram, count in ref_ngrams.items():
                        max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

                clipped_total += sum(min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items())

            precisions.append((clipped_total + 1e-9) / (hypothesis_total + 1e-9))

        hyp_length = sum(len(hyp) for hyp in hypotheses)
        ref_length = 0
        for refs, hyp in zip(references, hypotheses):
            hyp_len = len(hyp)
            if not refs:
                continue
            ref_lengths = [len(ref) for ref in refs]
            ref_length += min(ref_lengths, key=lambda length: (abs(length - hyp_len), length))

        if hyp_length == 0:
            return 0.0

        brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1 - (ref_length / hyp_length))
        score = brevity_penalty * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
        return float(score)

    def __mean_rouge_l(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> float:
        scores: List[float] = []
        for refs, hyp in zip(references, hypotheses):
            if not refs or not hyp:
                scores.append(0.0)
                continue
            scores.append(max(self.__rouge_l_f1(ref, hyp) for ref in refs))
        return sum(scores) / max(1, len(scores))

    def __rouge_l_f1(self, reference: List[str], hypothesis: List[str]) -> float:
        lcs = self.__lcs_length(reference, hypothesis)
        if lcs == 0:
            return 0.0
        precision = lcs / len(hypothesis)
        recall = lcs / len(reference)
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def __lcs_length(a: Iterable[str], b: Iterable[str]) -> int:
        a_list = list(a)
        b_list = list(b)
        prev = [0] * (len(b_list) + 1)
        for token_a in a_list:
            curr = [0]
            for j, token_b in enumerate(b_list, start=1):
                if token_a == token_b:
                    curr.append(prev[j - 1] + 1)
                else:
                    curr.append(max(prev[j], curr[-1]))
            prev = curr
        return prev[-1]

    @torch.inference_mode()
    def generate_caption(self, image_path: str, max_new_tokens: int = 32) -> str:
        image = Image.open(image_path).convert("RGB")
        pixels = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        img_raw = self.clip_model.encode_image(pixels)
        image_embed = self.projector(img_raw)

        inputs_embeds = image_embed.unsqueeze(1)
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_id = int(next_token_id.item())

            if self.llm_tokenizer.eos_token_id is not None and token_id == self.llm_tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            next_embed = self.llm_model.get_input_embeddings()(next_token_id)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            next_attention = torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

        return self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def evaluate(
        self,
        max_new_tokens: int = 32,
        image_root: Optional[str] = None,
        token_path: Optional[str] = None,
        split_path: Optional[str] = None,
    ) -> Dict:
        eval_image_root = image_root or self.config.image_path
        eval_token_path = token_path or self.config.text_path
        eval_split_path = split_path or self.config.test_split_path

        image_to_refs = self.__load_references(eval_token_path, eval_split_path)
        image_names = sorted(image_to_refs.keys())

        predictions: List[Dict] = []
        hypothesis_tokens: List[List[str]] = []
        reference_tokens: List[List[List[str]]] = []

        pbar = tqdm(image_names, desc="caption test", total=len(image_names))
        for image_name in pbar:
            image_path = str(Path(eval_image_root) / image_name)
            generated = self.generate_caption(image_path, max_new_tokens=max_new_tokens)
            references = image_to_refs[image_name]

            predictions.append(
                {
                    "image": image_name,
                    "prediction": generated,
                    "references": references,
                }
            )
            hypothesis_tokens.append(self.__tokenize_text(generated))
            reference_tokens.append([self.__tokenize_text(ref) for ref in references])

        metrics = {
            "num_samples": len(predictions),
            "bleu_1": self.__corpus_bleu(reference_tokens, hypothesis_tokens, [1.0, 0.0, 0.0, 0.0]),
            "bleu_2": self.__corpus_bleu(reference_tokens, hypothesis_tokens, [0.5, 0.5, 0.0, 0.0]),
            "bleu_3": self.__corpus_bleu(reference_tokens, hypothesis_tokens, [1 / 3, 1 / 3, 1 / 3, 0.0]),
            "bleu_4": self.__corpus_bleu(reference_tokens, hypothesis_tokens, [0.25, 0.25, 0.25, 0.25]),
            "rouge_l": self.__mean_rouge_l(reference_tokens, hypothesis_tokens),
        }
        return {"metrics": metrics, "predictions": predictions}

    def test(self, test_config: TestConfig) -> None:
        if test_config.image_path:
            caption = self.generate_caption(test_config.image_path, max_new_tokens=test_config.max_new_tokens)
            self.logger.info(json.dumps({"image": test_config.image_path, "caption": caption}, ensure_ascii=False))
            return

        result = self.evaluate(
            max_new_tokens=test_config.max_new_tokens,
            image_root=test_config.eval_image_path,
            token_path=test_config.eval_text_path,
            split_path=test_config.eval_split_path,
        )
        output_path = Path(test_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info(json.dumps(result["metrics"], ensure_ascii=False))
        self.logger.info(f"evaluation results saved to {output_path}")
