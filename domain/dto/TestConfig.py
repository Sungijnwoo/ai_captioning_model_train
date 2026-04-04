from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class TestConfig:
    model_dir: str
    llm_name: str
    projector_name: str
    image_path: Optional[str]
    eval_image_path: Optional[str]
    eval_text_path: Optional[str]
    eval_split_path: Optional[str]
    max_new_tokens: int
    output_path: str

    def __init__(self, data: Dict):
        self.__dict__.update(**data)

    @staticmethod
    def load_config(path: str) -> "TestConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)
        return TestConfig(data)

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
