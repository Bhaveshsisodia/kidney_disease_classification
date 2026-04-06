from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.components.model_training import Training
from kidney_disease_classifier import logger

STAGE_NAME = "Model Training Stage"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        logger.info(f"Training config: {training_config}")
        trainer = Training(config=training_config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e