from src import logger
from src.components.data_transformation import box_cox_transformation, min_max_scale, remove_zeros_in_df, save_preprocessed_excel
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_data_transformation_config

def data_transformation_training_pipeline():
    
    try:
        # Load config and schema
        config, _ = load_configuration()

        # Retrieve the data ingestion configuration from the loaded config
        data_transformation_config = get_data_transformation_config(config)

        # Create directories related to data ingestion (root directory)
        create_directories([data_transformation_config.root_dir])

        # # Box_cox transform
        box_cox = box_cox_transformation(data_transformation_config)
        
        # min max scale 
        min_max = min_max_scale(box_cox)

        # remove zeros
        remove_zeros=remove_zeros_in_df(min_max)

        # saving the file 
        save_preprocessed_excel(data_transformation_config, remove_zeros)

    except Exception as e:
        logger.error(f"An error occurred during data validation pipeline: {e}")

if __name__ == "__main__":

 data_transformation_training_pipeline()