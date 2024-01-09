from src.LCIC.constants import *
from src.LCIC.utils.common import read_yaml, create_directories
from src.LCIC.entity.data_config_entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(self, 
            config_path:Path = CONFIG_FILE_PATH, 
            params_path : Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_cfg = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            local_data_file= config.local_data_file
        )
        return data_ingestion_cfg