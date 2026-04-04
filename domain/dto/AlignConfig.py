from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class AlignConfig:
    clip_model_id: str
    clip_model_path: str
    llm_model_id: str
    project_multi_rate: int | None
    image_path: str
    text_path: str
    batch_size: int
    num_worker: int
    max_text_length: int
    align_epochs: int
    align_lr: float
    align_temperature: float
    output_dir: str

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
