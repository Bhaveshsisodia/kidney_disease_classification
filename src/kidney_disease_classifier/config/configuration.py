from kidney_disease_classifier.constants import *
import os
from kidney_disease_classifier import logger
from kidney_disease_classifier.utils.common import read_yaml, create_directories
from kidney_disease_classifier.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig , TrainingConfig
                                           )


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])



    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        logger.info(f"DataIngestionConfig: {data_ingestion_config}")


        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_weights = self.params.WEIGHTS,
            params_include_top = self.params.INCLUDE_TOP,
            params_classes = self.params.CLASSES,
            params_learning_rate = self.params.LEARNING_RATE
        )

        logger.info(f"PrepareBaseModelConfig: {prepare_base_model_config}")

        return prepare_base_model_config



    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")

        create_directories([training_config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE
        )
        logger.info(f"TrainingConfig: {training_config}")

        return training_config

