from LCIC import logger
from LCIC.config.model_eval_configuration import EvalConfigurationManager
from LCIC.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model evaluation stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):

        try:
            config_obj = EvalConfigurationManager()
            eval_config_obj = config_obj.get_evaluation_config()
            evaluation_obj = ModelEvaluation(config=eval_config_obj)
            evaluation_obj.run_evaluation()
            evaluation_obj.log_into_mlflow()
        except Exception as e:
            logger.error(f"Occured error in stage-2: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"*******************************************")
        logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} <<<<<<<<<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.run()
        logger.info(
            f">>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<\n\nx========================x\n\n")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME} \n {e}")
        raise e
