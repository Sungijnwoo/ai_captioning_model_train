from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    clip_model: str
    clip_model_path: str
    llm_model: str
    image_path: str
    text_path: str
    project_multi_rate: int
    batch_size: int
    num_worker: int
    max_text_length: int
    align_epochs: int
    align_temperature: float
    caption_epochs: int
    align_lr: float
    caption_lr: float
    train_samples: int
    eval_samples: int
    output_dir: str

    @staticmethod
    def load_config(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)
        return Config(**data)

    def __repr__(self) -> str:
        data = asdict(self)
        return yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=4)
