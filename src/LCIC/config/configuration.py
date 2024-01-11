from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, create_directories
from LCIC.entity.data_config_entity import DataIngestionConfig
from LCIC.entity.data_prep_entity import PreprocessConfig


class ConfigurationManager:
    def __init__(self,
                 config_path: Path = CONFIG_FILE_PATH,
                 params_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.artifacts_root])

    # this the method to get the data ingestion configuration for the stage-1: data ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_cfg = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file
        )
        return data_ingestion_cfg

    # this the method to get the data preprocessing configuration for the stage-2: data preprocessing
    def get_preprocess_config(self) -> PreprocessConfig:

        preprocessed_cfg = self.config.data_preporcess

        logger.info(f"Preprocessing configurations: ---> {preprocessed_cfg}")
        _cfg = PreprocessConfig(
            # locally downloade data path
            path=input(
                "Enter the data path: ") if self.config.data_ingestion.local_data_file is None else self.config.data_ingestion.local_data_file,
            rescale=preprocessed_cfg.rescale,
            shear_range=preprocessed_cfg.shear_range,
            zoom_range=preprocessed_cfg.zoom_range,
            width_shift_range=preprocessed_cfg.width_shift_range,
            height_shift_range=preprocessed_cfg.height_shift_range,
            horizontal_flip=preprocessed_cfg.horizontal_flip,
            validation_split=preprocessed_cfg.validation_split,
            fill_mode=preprocessed_cfg.fill_mode,
            batch_size=preprocessed_cfg.batch_size,
            seed=preprocessed_cfg.seed
        )
        return _cfg
