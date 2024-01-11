from LCIC import logger
from LCIC.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from LCIC.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from LCIC.pipeline.stage_03_model_building import ModelBuildingPipeline

if __name__ == "__mian__":
    try:
        STAGE_NAME = "Data ingestion stage"
        logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataIngestionPipeline()

        logger.info(f"Downloading data from internet")
        obj.run()

        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

        STAGE_NAME = "Data Preprocessing stage"
        logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataPreprocessingPipeline()

        logger.info(f"Preprocessing data for training and validation set")
        # if you want to get train and valid set then set it to True
        obj.run(get_train_and_valid_set=False)

        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

        STAGE_NAME = "Data Preprocessing stage"
        logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataPreprocessingPipeline()

        logger.info(f"Preprocessing data for training and validation set")
        # if you want to get train and valid set then set it to True
        obj.run(get_train_and_valid_set=False)

        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

        STAGE_NAME = "Model Building stage"
        logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelBuildingPipeline()

        logger.info(f"Preprocessing data for training and validation set")
        # if you want to get train and valid set then set it to True
        obj.run()

        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

    except Exception as e:
        logger.error(e)
        raise e
