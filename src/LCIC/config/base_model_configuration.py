from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, create_directories

from LCIC.entity.base_model_entity import BaseModelConfig


class BaseModelConfigurationManager:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.artifacts_root])

    def get_base_model_config(self) -> BaseModelConfig:
        base_model_cfg = self.config.base_model
        self.model_name = base_model_cfg.model_name
        self.params = self.params[base_model_cfg.model_name]

        logger.info(f"Configurations ---> {base_model_cfg}")
        logger.info(f"Parameters ---> {self.params}")

        create_directories([base_model_cfg.root_dir])

        _cfg = BaseModelConfig(
            pretrained_model_name=self.model_name,
            root_dir= Path(base_model_cfg.root_dir),
            base_model_path= Path(base_model_cfg.base_model_path),
            updated_base_model_path= Path(base_model_cfg.updated_base_model_path),
            params_include_top= self.params.INCLUDE_TOP,
            params_weights= self.params.WEIGHTS,
            params_image_size= self.params.IMAGE_SIZE,
            params_optimizer= self.params.OPTIMIZER,
            params_learning_rate= self.params.LEARNING_RATE,
            params_dropout_rate= self.params.DROPOUT_RATE,
            params_classes= self.params.CLASSES  # need to look at.....!
        )
        return _cfg
