from src import logger
from src.pipeline.stage_01_data_ingestion import data_ingestion_training_pipeline
from src.pipeline.stage_02_data_validation import data_validation_training_pipeline
from src.pipeline.stage_03_data_preprocessing import data_preprocessing_training_pipeline
from src.pipeline.stage_04_data_transformer import data_transformation_training_pipeline
from src.pipeline.stage_05_model_trainer import model_trainer_training_pipeline
from src.pipeline.stage_06_model_evaluation import model_evaluation_training_pipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion=data_ingestion_training_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation=data_validation_training_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Data Preprocessing stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_preprocess=data_preprocessing_training_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation=data_transformation_training_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Model Training stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer= model_trainer_training_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)

# STAGE_NAME = "Model Evaluation stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     model_evaluation= model_evaluation_training_pipeline()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)

