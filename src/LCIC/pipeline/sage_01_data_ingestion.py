from src.LCIC import logger
from src.LCIC.components.data_ingestion import DataIngestion
from src.LCIC.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        
        try:
            config_obj = ConfigurationManager()
            data_ingestion_config_obj = config_obj.get_data_ingestion_config()
            data_ingestion_obj = DataIngestion(config=data_ingestion_config_obj)
            
            data_ingestion_obj.get_data_ingestion_config()
        except Exception as e:
            logger.error(e)
            raise e

