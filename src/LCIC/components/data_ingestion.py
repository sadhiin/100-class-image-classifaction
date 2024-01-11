from LCIC import logger
import opendatasets as od
from LCIC.entity.data_ingestion_config import DataIngestionConfig
class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Fetch the specified dataset from internet and store the dataset locally.
        return: None
        """

        try:
            download_url = self.config.source_url
            save_at = self.config.local_data_file
            logger.info(f"Downloading dataset form {download_url}")
            od.download(dataset_id_or_url=download_url, data_dir=save_at,dry_run=True)
            logger.info(f"Successfully saved at {save_at}")
        except Exception as e:
            logger.error(f"Error in downloading data. {e}")
            raise e
