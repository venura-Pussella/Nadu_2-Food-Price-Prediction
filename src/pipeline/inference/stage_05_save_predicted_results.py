from src.components.inference.inference_stage_04_predictions_store import save_predictions_to_csv 
from src.configuration.configuration import load_configuration, get_prediction_results_config
from src.utils.common import create_directories

def save_predictions(predicted_df):

    # Load the configuration
    config, _ = load_configuration()

    prediction_results_config = get_prediction_results_config(config)

    # Create directories related to model trainer (root directory)
    create_directories([prediction_results_config.root_dir])

    save_predictions_to_csv(predicted_df,prediction_results_config)

    return save_predictions_to_csv
