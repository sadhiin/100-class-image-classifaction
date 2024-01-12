from src.LCIC import logger
from src.LCIC.config.model_training_configureation import ConfigurationManager
from LCIC.components.model_training import Training


STAGE_NAME = "Data Preprocessing and Model training stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):

        try:
            config_obj = ConfigurationManager()
            train_config_obj = config_obj.get_model_train_config()
            traning_obj = Training(config=train_config_obj)
            traning_obj.train()

        except Exception as e:
            logger.error(f"Occured error in stage-2: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"*******************************************")
        logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} <<<<<<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.run()
        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME} \n {e}")
        raise e
