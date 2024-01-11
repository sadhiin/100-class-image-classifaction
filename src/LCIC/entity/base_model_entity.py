from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaseModelConfig:
    pretrained_model: str
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_include_top: bool
    params_weights: str
    params_image_size: list
    params_learning_rate: float
    params_optimizer: str
    params_dropout_rate: float
    params_classes: int
