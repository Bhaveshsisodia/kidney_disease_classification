from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.components.mode_evaluation_mlflow import Evaluation
from kidney_disease_classifier import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            logger.info(f"Evaluation config: {evaluation_config}")
            evaluation = Evaluation(config=evaluation_config)
            evaluation.evaluate()
            evaluation.log_into_mlflow()
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
