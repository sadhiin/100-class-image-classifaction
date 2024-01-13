from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_path: Path
    trained_model_path: Path
    model_name: str

    dataset_path: str
    batch_size: int
    seed: int
    shear_range: float
    zoom_range: float
    width_shift_range: float
    height_shift_range: float
    horizontal_flip: bool
    validation_split: float
    fill_mode: str
