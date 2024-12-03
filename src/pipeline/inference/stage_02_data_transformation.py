from src.configuration.configuration import load_configuration, get_data_transformation_config
from src.utils.common import read_lambda_value
from src.components.inference.inference_stage_02_data_transformation import box_cox_with_min_max_scaling

def inference_data_transformation(current_sequence):    

    # Load the configuration
    config, _ = load_configuration()

    # Retrieve data ingestion configuration from the loaded config
    data_transformation_config = get_data_transformation_config(config)

    # Read the lambda value
    lambda_value = read_lambda_value(data_transformation_config)

    # input the sequences to Box_cox transform and Min_Max_Scaling 
    transformed_prices = box_cox_with_min_max_scaling(current_sequence, lambda_value)

    return transformed_prices

if __name__ == "__main__":

 inference_data_transformation()