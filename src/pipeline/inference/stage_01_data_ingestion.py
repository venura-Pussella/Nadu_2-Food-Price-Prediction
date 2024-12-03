from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query_30_day
from src.components.inference.inference_stage_01_data_ingestion_latest_30days import fetch_cosmosdb_data_to_dataframe_latest30days, process_cosmosdb_dataframe_latest30days

def inference_data_ingestion():

    # Fetch data from Cosmos DB
    df = fetch_cosmosdb_data_to_dataframe_latest30days(cosmos_url, cosmos_key, database_name, container_name, query_30_day)

    df, current_sequence , last_date = process_cosmosdb_dataframe_latest30days(df)

    return df, current_sequence , last_date

if __name__ == "__main__":

    df, current_sequence, last_date = inference_data_ingestion()
    print(df, current_sequence, last_date) 
