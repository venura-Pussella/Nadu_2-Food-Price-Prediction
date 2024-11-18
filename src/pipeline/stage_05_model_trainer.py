from src import logger
from src.components.model_trainer import sequence_creation_with_forecast, save_train_test_data_to_excel, lstm_model_trainer
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_model_trainer_config

from src import logger

def model_trainer_training_pipeline():
    try:
        # Load config and schema
        config,schema = load_configuration()

        # Retrieve the model trainer configuration from the loaded config
        model_trainer_config = get_model_trainer_config(config,schema)

        # Create directories related to model trainer (root directory)
        create_directories([model_trainer_config.root_dir, model_trainer_config.root_dir_train, model_trainer_config.root_dir_test])

        # Sequence creation
        train_x, test_x, train_y, test_y, train_dates, test_dates = sequence_creation_with_forecast(model_trainer_config)

        print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
        
        # Saving data to an excel from numpy array (3D to 2D)
        save_train_test_data_to_excel(train_x, test_x, train_y, test_y,train_dates, test_dates, model_trainer_config)

        # Train the model
        lstm_model_trainer(train_x, train_y, model_trainer_config)

    except Exception as e:
        logger.error(f"An error occurred in model trainer : {e}")

if __name__ == "__main__":

    model_trainer_training_pipeline()