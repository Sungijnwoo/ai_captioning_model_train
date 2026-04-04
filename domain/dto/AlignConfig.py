from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AlignConfig:
    clip_model_id: str
    clip_model_path: str
    llm_model_id: str
    project_multi_rate: Optional[int]
    image_path: str
    text_path: str
    train_split_path: Optional[str] = None
    val_split_path: Optional[str] = None
    test_split_path: Optional[str] = None
    batch_size: int = 8
    num_worker: int = 4
    max_text_length: int = 48
    align_epochs: int = 1
    align_lr: float = 1.0e-4
    align_temperature: float = 0.07
    early_stop_patience: int = 0
    output_dir: str = "output/align"

    @staticmethod
    def load_config(path: str) -> "AlignConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)
        return AlignConfig(**data)

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
