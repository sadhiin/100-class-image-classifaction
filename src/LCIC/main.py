from LCIC import logger
from LCIC.pipeline.sage_01_data_ingestion import DataIngestionPipeline

if __name__=="__mian__":
    try:
        STAGE_NAME = "Data ingestion stage"
        logger.info(f">>>>>>>>> At stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataIngestionPipeline()
       
        logger.info(f"Downloading data from internet")
        obj.run()
    
        logger.info(f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
    except Exception as e:
        logger.error(e)
        raise e