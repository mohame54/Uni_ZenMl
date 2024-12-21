
from steps import serve_model, download_convert_model
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def serve_model_mlflow():
    logger.info("Creating a pipeline to serve the model using MLFlow")
    download_convert_model()
    serve_model()
