from src import logger
from src.components.training.model_trainer import sequence_creation_with_forecast, sequence_creation_with_forecast_full_train, save_train_test_data_model_evaluation, lstm_model_trainer,save_full_training_data_to_excel,lstm_full_model_trainer
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_model_trainer_config

def model_trainer_training_pipeline():
    try:
        # Load config and schema
        config,schema = load_configuration()

        # Retrieve the model trainer configuration from the loaded config
        model_trainer_config = get_model_trainer_config(config,schema)

        # Create directories related to model trainer (root directory)
        create_directories([model_trainer_config.root_dir, model_trainer_config.root_dir_train, model_trainer_config.root_dir_test])

        # Sequence creation
        train_x, train_y,test_x, test_y, train_x_dates, train_y_dates,test_x_dates, test_y_dates = sequence_creation_with_forecast(model_trainer_config)

        # Saving data to an excel from numpy array (3D to 2D)
        save_train_test_data_model_evaluation(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            train_x_dates=train_x_dates,
            train_y_dates=train_y_dates,
            test_x_dates=test_x_dates,
            test_y_dates=test_y_dates,
            config=model_trainer_config,
        )

        # Train the model
        lstm_model_trainer(train_x, train_y, model_trainer_config)

    except Exception as e:
        logger.error(f"An error occurred in model trainer : {e}")


def model_full_trainer_training_pipeline():
    try:
        # Load config and schema
        config,schema = load_configuration()

        # Retrieve the model trainer configuration from the loaded config
        model_trainer_config = get_model_trainer_config(config,schema)

        # Create directories related to model trainer (root directory)
        create_directories([model_trainer_config.root_dir, model_trainer_config.root_dir_train, model_trainer_config.root_dir_test])

        # Sequence creation
        train_x, train_y, train_x_dates, train_y_dates = sequence_creation_with_forecast_full_train(model_trainer_config)

        # Saving data to an excel from numpy array (3D to 2D)
        save_full_training_data_to_excel(train_x, train_y, train_x_dates, train_y_dates, model_trainer_config)

        # Train the model
        lstm_full_model_trainer(train_x, train_y, model_trainer_config)

    except Exception as e:
        logger.error(f"An error occurred in model trainer : {e}")


if __name__ == "__main__":

    model_trainer_training_pipeline()