from src import logger
import tensorflow as tf
from src.pipeline.inference.inference_stage_preprocessing import box_cox_with_min_max_scaling
from src.pipeline.inference.inference_stage_output_preprocessing import inverse_transform_output

def prediction_pipeline(input_prices, lambda_value):
    
    try:

        # input the sequences to Box_cox transform and Min_Max_Scaling 
        transformed_prices = box_cox_with_min_max_scaling(input_prices, lambda_value)

        # Load your trained LSTM model
        model =  tf.keras.models.load_model('artifacts/model_trainer/model/5day_model.keras')

        # make prediction
        prediction = model.predict(transformed_prices)

        print(prediction)

        # Inverse transform the prediction to see real values
        true_price = inverse_transform_output(prediction, lambda_value)

        return true_price
    
    except Exception as e:
        logger.error(f"An error occurred during inference pipeline: {e}")