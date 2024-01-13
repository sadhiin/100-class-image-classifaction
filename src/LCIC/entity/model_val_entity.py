from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EvaluationConfig:
    model_name: str
    model_path: Path
    dataset_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int