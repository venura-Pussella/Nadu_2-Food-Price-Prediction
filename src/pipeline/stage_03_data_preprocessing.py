from src import logger
from src.components.data_preprocessing import data_read_clean_missing_values, drop_unnecessary_columns, save_preprocessed_excel
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_data_preprocessing_config

def data_preprocessing_training_pipeline():
    
    try:
        # Load config and schema
        config, _ = load_configuration()

        # Retrieve the data ingestion configuration from the loaded config
        data_preprocessing_config = get_data_preprocessing_config(config)

        # Create directories related to data ingestion (root directory)
        create_directories([data_preprocessing_config.root_dir])

        cleaned_data = data_read_clean_missing_values(data_preprocessing_config)

        dropped_columns_data=drop_unnecessary_columns(cleaned_data)

        save_preprocessed_excel(data_preprocessing_config, dropped_columns_data)

    except Exception as e:
        logger.error(f"An error occurred during data preprocessing pipeline: {e}")

if __name__ == "__main__":

 data_preprocessing_training_pipeline()