from src import logger
from src.components.training.training_data_validation import validate_all_columns
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_data_validation_config


def data_validation_training_pipeline():
    """Runs the data validation pipeline."""
    try:
        # Load config and schema
        config , schema = load_configuration()

        # Retrieve the data ingestion configuration from the loaded config
        data_validation_config = get_data_validation_config(config, schema)

        # Create directories related to data ingestion (root directory)
        create_directories([data_validation_config.root_dir])

        # Download the data file as part of the data ingestion process
        validate_all_columns(data_validation_config)

    except Exception as e:
        logger.error(f"An error occurred during data validation pipeline: {e}")

if __name__ == "__main__":

    data_validation_training_pipeline()