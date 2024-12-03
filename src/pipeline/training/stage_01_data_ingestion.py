from src import logger
from src.components.training.training_data_ingestion import fetch_cosmosdb_data_to_dataframe , preprocess_dataframe
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_data_ingestion_config
from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query

def data_ingestion_training_pipeline():

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
        df = fetch_cosmosdb_data_to_dataframe(cosmos_url, cosmos_key, database_name, container_name, query)

        preprocess_dataframe(df, data_ingestion_config)

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {e}")

if __name__ == "__main__":

 data_ingestion_training_pipeline()
