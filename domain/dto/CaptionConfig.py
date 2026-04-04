from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import yaml

from domain.dto.AlignConfig import AlignConfig


@dataclass
class CaptionConfig:
    project_model_dir: str
    caption_epochs: int
    caption_lr: float
    batch_size: 8
    num_worker: 4
    output_dir: str

    clip_model_id: str
    clip_model_path: str
    llm_model_id: str
    project_multi_rate: int
    image_path: str
    text_path: str
    max_text_length: int

    def __init__(self, data: Dict):
        self.__dict__.update(**data)
        align_config = AlignConfig.load_config(f"{self.project_model_dir}/align_config.yaml")
        self.clip_model_id = align_config.clip_model_id
        self.clip_model_path = align_config.clip_model_path
        self.llm_model_id = align_config.llm_model_id
        self.project_multi_rate = align_config.project_multi_rate
        self.image_path = align_config.image_path
        self.text_path = align_config.text_path
        self.max_text_length = align_config.max_text_length

    @staticmethod
    def load_config(path: str) -> "CaptionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)
        return CaptionConfig(data)

    def __repr__(self) -> str:
        data = asdict(self)
        return yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=4)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                asdict(self),
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                indent=4,
            )
