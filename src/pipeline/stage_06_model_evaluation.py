from src import logger
from src.configuration.configuration import get_model_trainer_config , get_model_evaluation_config
from src.components.model_trainer import sequence_creation_with_forecast
from src.components.model_evaluation import model_evaluation
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration

def model_evaluation_training_pipeline():
    
    try:
        # Load config and schema
        config, schema = load_configuration()

        model_trainer_config = get_model_trainer_config(config,schema) 

        # Retrieve the model trainer configuration from the loaded config
        model_evaluation_config = get_model_evaluation_config(config)

        # Create directories related to model evaluation (root directory)
        create_directories([model_evaluation_config.root_dir])

        train_x, train_y, test_x, test_y, train_x_dates, train_y_dates, test_x_dates, test_y_dates = sequence_creation_with_forecast(model_trainer_config)

        # model evaluation
        model_evaluation(train_x, test_x, train_y, test_y,model_evaluation_config)

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")

if __name__ == "__main__":

 model_evaluation_training_pipeline()