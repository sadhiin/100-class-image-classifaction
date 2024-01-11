from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    path: str
    rescale: float
    shear_range: float
    zoom_range: float
    width_shift_range: float
    height_shift_range: float
    horizontal_flip: bool
    validation_split: float
    fill_mode: str
    batch_size: int
    seed: int
