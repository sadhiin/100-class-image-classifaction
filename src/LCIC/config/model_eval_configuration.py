from LCIC.constants import *
from LCIC.utils.common import read_yaml
from LCIC.entity.model_val_entity import EvaluationConfig


class EvalConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_file_path)

    def get_evaluation_config(self) -> EvaluationConfig:

        self.config = self.config.model_eval
        self.params = self.params[self.config.model_name]

        eval_cfg = EvaluationConfig(
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            dataset_path=self.config.dataset_path,
            all_params=self.params,
            mlflow_uri=self.config.mlflow_uri,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_cfg
