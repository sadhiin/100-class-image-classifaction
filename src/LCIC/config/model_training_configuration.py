from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, create_directories
from LCIC.entity.model_training_entity import ModelTrainingConfig


class ConfigurationManager:
    def __init__(self,
                 config_path: Path = CONFIG_FILE_PATH,
                 params_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.model_training.root_dir])

    def get_model_train_config(self) -> ModelTrainingConfig:

        model_cfg = self.config.model_training
        preprocessed_cfg = self.config.model_training.data_processing
        self.params = self.params[model_cfg.model_name]

        logger.info(f"Model configurations: ---> {model_cfg}")
        logger.info(f"Model parameters: ---> {preprocessed_cfg}")
        logger.info(
            f"Data Preprocessing configurations: ---> {preprocessed_cfg}")

        _cfg = ModelTrainingConfig(
            root_dir=Path(model_cfg.root_dir),
            model_path=Path(model_cfg.model_path),
            trained_model_path=Path(model_cfg.trained_model_path),
            model_name=model_cfg.model_name,

            dataset_path=input("Enter the data path: ") if preprocessed_cfg.dataset_path is None or preprocessed_cfg.dataset_path=='' else preprocessed_cfg.dataset_path,
            batch_size=preprocessed_cfg.batch_size,
            seed=preprocessed_cfg.seed,
            shear_range=preprocessed_cfg.shear_range,
            zoom_range=preprocessed_cfg.zoom_range,
            width_shift_range=preprocessed_cfg.width_shift_range,
            height_shift_range=preprocessed_cfg.height_shift_range,
            horizontal_flip=preprocessed_cfg.horizontal_flip,
            validation_split=preprocessed_cfg.validation_split,
            fill_mode=preprocessed_cfg.fill_mode,
        )
        return _cfg
