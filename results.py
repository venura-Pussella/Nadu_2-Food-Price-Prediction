from src.configuration.configuration import load_configuration,get_prediction_results_config
from src.components.inference.inference_results_writer import update_real_values_in_csv

def results():

    # Load the configuration
    config, _ = load_configuration()

    # Retrieve the data ingestion configuration from the loaded config
    prediction_results_config = get_prediction_results_config(config)

    update_real_values_in_csv(prediction_results_config)

if __name__ == '__main__':
     
     results()


