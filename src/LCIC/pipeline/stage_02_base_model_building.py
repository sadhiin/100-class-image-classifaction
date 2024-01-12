from LCIC import logger
from LCIC.config.base_model_configuration import BaseModelConfigurationManager
from LCIC.components.model_builder import PrepareBaseModel

STAGE_NAME = "Model Building stage"


class ModelBuildingPipeline():

    def __init__(self):
        pass

    def run(self):

        try:
            config_manager = BaseModelConfigurationManager()
            base_model_cfg_obj = config_manager.get_base_model_config()

            base_model_creator_obj = PrepareBaseModel(config=base_model_cfg_obj)  

            logger.info(f"Creating base model")
            base_model_creator_obj.get_base_model()

            logger.info(f"Updating base model with custome layers.")
            base_model_creator_obj.update_base_model()
            
            logger.info(f"Base model created successfully.")
        except Exception as e:
            logger.error("Error in creating base model")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"*******************************************")
        logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} <<<<<<<<<<<<<<<")
        obj = ModelBuildingPipeline()
        obj.run()
        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME} \n {e}")
        raise e
