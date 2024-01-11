from src.LCIC import logger
from src.LCIC.config.configuration import ConfigurationManager
from src.LCIC.components.data_preprocess import DataPreprocessing

STAGE_NAME = "Data Preprocessing stage"


class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def run(self, get_train_and_valid_set=False):

        try:
            config_obj = ConfigurationManager()
            data_preprocess_config_obj = config_obj.get_preprocess_config()
            data_preprocessing_obj = DataPreprocessing(
                config=data_preprocess_config_obj)
            logger.info(
                f"Loadding dataset for training & validation set with preprocessing using TF.")
            training_set, validation_set = data_preprocessing_obj.get_train_and_valid_set()
            if get_train_and_valid_set:
                return training_set, validation_set
        except Exception as e:
            logger.error(f"Occured error in stage-2: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"*******************************************")
        logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} <<<<<<<<<<<<<<<")
        obj = DataPreprocessingPipeline()
        obj.run()
        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME} \n {e}")
        raise e
