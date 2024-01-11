from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, create_directories

from LCIC.entity.base_model_entity import BaseModelConfig


class BaseModelConfigurationManager:
    def __init__(self,
                 model_name: str = "",
                 config_path: Path = CONFIG_FILE_PATH,
                 params_path: Path = PARAMS_FILE_PATH):
        if model_name == "" or model_name is None:
            model_name = input("Enter the model name (all lower-case): ")
        # for selecting different pretrained models
        self.model_name = model_name.upper()
        self.config = read_yaml(config_path)
        # only the specified model configuration it will take.
        self.params = read_yaml(params_path)[self.model_name]

        create_directories([self.config.artifacts_root])
        logger.info(
            f"Preparing config manager for {self.model_name} pretrained model.")

    def get_base_model_config(self) -> BaseModelConfig:
        base_model_cfg = self.config.base_model

        logger.info(f"Configurations ---> {base_model_cfg}")
        logger.info(f"Parameters ---> {self.params}")

        create_directories([base_model_cfg.root_dir])

        _cfg = BaseModelConfig(
            pretrained_model=self.model_name,
            root_dir=Path(base_model_cfg.root_dir),
            base_model_path=Path(base_model_cfg.base_model_path),
            updated_base_model_path=Path(
                base_model_cfg.updated_base_model_path),
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_image_size=self.params.IMAGE_SIZE,
            params_optimizer=self.params.OPTIMIZER,
            params_learning_rate=self.params.LEARNING_RATE,
            params_dropout_rate=self.params.DROPOUT_RATE,
            params_classes=self.params.CLASSES  # need to look at.....!
        )
        return _cfg
