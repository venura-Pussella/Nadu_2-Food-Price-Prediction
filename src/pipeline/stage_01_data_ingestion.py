from src import logger
from src.components.data_ingestion import download_file
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_data_ingestion_config

def data_ingestion_training_pipeline():
    """Runs the data ingestion pipeline."""
    try:
        # Load config and schema
        config , _ = load_configuration()

        # Create directories for artifacts root
        create_directories([config.artifacts_root])

        # Retrieve the data ingestion configuration from the loaded config
        data_ingestion_config = get_data_ingestion_config(config)

        # Create directories related to data ingestion (root directory)
        create_directories([data_ingestion_config.root_dir])

        # Download the data file as part of the data ingestion process
        download_file(data_ingestion_config)

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {e}")

if __name__ == "__main__":

 data_ingestion_training_pipeline()
