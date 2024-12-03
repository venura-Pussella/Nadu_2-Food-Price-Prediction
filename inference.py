from src import logger
from src.pipeline.inference.stage_01_data_ingestion import inference_data_ingestion
from src.pipeline.inference.stage_02_data_transformation import inference_data_transformation
from src.pipeline.inference.stage_03_make_predictions import make_predictions
from src.pipeline.inference.stage_04_data_inverse_transform import data_inverse_transform
from src.pipeline.inference.stage_05_save_predicted_results import save_predictions

STAGE_NAME = "Inference Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    df, current_sequence , last_date =inference_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    print(current_sequence)
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Inference Data Transformation"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    transformed_prices =inference_data_transformation(current_sequence)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    print(transformed_prices.shape)
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Inference Make Predictions"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    predictions = make_predictions(transformed_prices)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    print(predictions)
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Inference Data Inverse Transform"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    predicted_df= data_inverse_transform(predictions,last_date)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    print(predicted_df)
except Exception as e:
    logger.exception(e)


STAGE_NAME = "Inference Save Predicted Results"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    results_csv =save_predictions(predicted_df)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    print(results_csv)
except Exception as e:
    logger.exception(e)