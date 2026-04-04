from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from domain.dto.AlignConfig import AlignConfig


@dataclass
class CaptionConfig:
    project_model_dir: str
    caption_epochs: int
    caption_lr: float
    early_stop_patience: int
    batch_size: int
    num_worker: int
    output_dir: str

    clip_model_id: str
    clip_model_path: str
    llm_model_id: str
    project_multi_rate: int
    image_path: str
    text_path: str
    max_text_length: int
    train_split_path: Optional[str]
    val_split_path: Optional[str]
    test_split_path: Optional[str]

    def __init__(self, data: Dict):
        self.__dict__.update(**data)
        align_config = AlignConfig.load_config(f"{self.project_model_dir}/align_config.yaml")
        self.clip_model_id = self.__resolve("clip_model_id", align_config.clip_model_id)
        self.clip_model_path = self.__resolve("clip_model_path", align_config.clip_model_path)
        self.llm_model_id = self.__resolve("llm_model_id", align_config.llm_model_id)
        self.project_multi_rate = self.__resolve("project_multi_rate", align_config.project_multi_rate)
        self.image_path = self.__resolve("image_path", align_config.image_path)
        self.text_path = self.__resolve("text_path", align_config.text_path)
        self.max_text_length = self.__resolve("max_text_length", align_config.max_text_length)
        self.train_split_path = self.__resolve("train_split_path", align_config.train_split_path)
        self.val_split_path = self.__resolve("val_split_path", align_config.val_split_path)
        self.test_split_path = self.__resolve("test_split_path", align_config.test_split_path)

    def __resolve(self, key: str, default):
        value = self.__dict__.get(key)
        return default if value is None else value

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
