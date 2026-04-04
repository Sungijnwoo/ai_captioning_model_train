import os
from typing import Tuple

import open_clip
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class Util:
    @staticmethod
    def load_clip_model(clip_model_path: str, clip_model_id: str) -> Tuple[torch.nn.Module, callable, callable]:
        pretrained = clip_model_path
        name = clip_model_id.split("/")[-1]
        model_kwargs = {}
        if not (name.endswith("S3") or name.endswith("S4") or name.endswith("L-14")):
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrained,
            weights_only=False,
            **model_kwargs,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        return model, preprocess_train, preprocess_val

    @staticmethod
    def load_llm_model(llm_model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id)
        llm_model.eval()
        for p in llm_model.parameters():
            p.requires_grad = False

        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        return llm_model, llm_tokenizer

    @staticmethod
    def download_clip_model(clip_model_id: str, clip_model_path: str):
        model_id = clip_model_id
        model_name = clip_model_path.split("/")[-1]
        local_folder_path = os.path.dirname(clip_model_path)

        hf_hub_download(
            repo_id=model_id,
            filename=model_name,
            local_dir=local_folder_path,
            local_dir_use_symlinks=False,
        )
