import pandas as pd
from azure.cosmos import CosmosClient

def fetch_cosmosdb_data_to_dataframe_latest30days(cosmos_url, cosmos_key, database_name, container_name, query_30_day):

    # Initialize the Cosmos client
    client = CosmosClient(cosmos_url, cosmos_key)

    # Get the database and container clients
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # Execute the query
    items = list(container.query_items(
        query=query_30_day,
        enable_cross_partition_query=True
    ))

    # Convert the results to a DataFrame
    df = pd.DataFrame(items)

    return df

def process_cosmosdb_dataframe_latest30days(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame but got {type(df)} instead.")

    # Drop the specified columns
    df = df.drop(columns=['id', 'page', '_rid', '_self', '_etag', '_attachments', '_ts', 'category', 'item', 'pettah_min_value', 'pettah_max_value'])
        
    # Convert 'date' column to datetime if it's not already in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sort the DataFrame by the 'date' column in ascending order
    df = df.sort_values(by='date', ascending=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Keep the first occurrence and drop the rest based on the 'date' column
    df = df.drop_duplicates(subset=['date'], keep='first')

    # Set the 'date' column as the index for interpolation
    df.set_index('date', inplace=True)

    # Interpolate missing values for 'pettah_average'
    df = df.resample('D').interpolate(method='linear')

    # Reset index to get 'date' back as a column
    df.reset_index(inplace=True)

    return df