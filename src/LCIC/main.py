from LCIC import logger
from LCIC.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from LCIC.pipeline.stage_02_base_model_building import ModelBuildingPipeline
from LCIC.pipeline.stage_03_model_training import ModelTrainingPipeline
from LCIc.pipeline.stage_04_model_eval import ModelEvaluationPipeline

try:
    # Stage -1 for data ingestion/downloading
    STAGE_NAME = "Data ingestion stage"
    logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
    obj = DataIngestionPipeline()

    logger.info(f"Downloading data from internet")
    obj.run()

    logger.info(
        f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

    # stage-2 for model building
    STAGE_NAME = "Model Building stage"
    logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
    obj = ModelBuildingPipeline()

    logger.info(f"Preprocessing data for training and validation set")
    obj.run()

    logger.info(
        f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

    # stage-3 for data processing and model training
    STAGE_NAME = "Data Preprocessing and Model Training stage"
    logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
    obj = ModelTrainingPipeline()

    logger.info(f"Preprocessing data for training and validation set")
    obj.run()

    logger.info(

        f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")

    # stage-4 model evaluation and logging the performance in dagshub with mlflow
    STAGE_NAME = "Model Evaluation stage"
    logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
    obj = ModelEvaluationPipeline()

    logger.info(f"Running model evaluation on the test set")
    obj.run()

    logger.info(

        f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
except Exception as e:
    logger.error(e)
    raise e
