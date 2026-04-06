import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from kidney_disease_classifier import logger
from kidney_disease_classifier.entity.config_entity import EvaluationConfig
from kidney_disease_classifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )





    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        logger.info(f"Loading model from path: {path}")
        model = tf.keras.models.load_model(path)
        logger.info("Model loaded successfully")
        return model

    def evaluate(self):
        logger.info("Starting model evaluation")
        self.train_valid_generator()
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"Evaluation results: {self.score}")
        self.save_score()

    def save_score(self):
        scores ={"loss": self.score[0] , "accuracy":self.score[1]}
        save_json(path =Path("scores.json"), data= scores)

    def log_into_mlflow(self):
        logger.info("Logging evaluation results into MLflow")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Kidney Disease Classification")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name="Evaluation") as mlops_run:
            logger.info("Logging parameters and metrics into MLflow")
            params = self.config.all_params.to_dict() if hasattr(self.config.all_params, "to_dict") else dict(self.config.all_params)
            mlflow.log_params(params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            if tracking_url_type_store != "file":
                logger.info("Logging model into MLflow Model Registry")
                mlflow.keras.log_model(self.model, "VGG16", registered_model_name="KidneyDiseaseClassificationModel")
            else:
                logger.info("Logging model into MLflow as artifact")
                mlflow.keras.log_model(self.model, "model")





