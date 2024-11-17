from box import ConfigBox
import pandas as pd
from azure.cosmos import CosmosClient

def fetch_cosmosdb_data_to_dataframe(cosmos_url, cosmos_key, database_name, container_name, query):

    # Initialize the Cosmos client
    client = CosmosClient(cosmos_url, cosmos_key)

    # Get the database and container clients
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # Execute the query
    items = list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    # Convert the results to a DataFrame
    df = pd.DataFrame(items)

    return df

def preprocess_dataframe(df: pd.DataFrame, config: ConfigBox):

    # Drop specified columns
    df = df.drop(columns=['id', 'page', '_rid', '_self', '_etag', '_attachments', '_ts',
                          'category', 'pettah_min_value', 'pettah_max_value'], errors='ignore')

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sort the DataFrame by 'date' column
    df = df.sort_values(by='date', ascending=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Drop duplicate dates, keeping the first occurrence
    df = df.drop_duplicates(subset=['date'], keep='first')

    # Set 'date' as the index for interpolation
    df.set_index('date', inplace=True)

    df = df.infer_objects(copy=False)

    # Resample to daily frequency and interpolate missing values
    df = df.resample('D').interpolate(method='linear')

    # Reset index to bring 'date' back as a column
    df.reset_index(inplace=True)

    # Ensure 'pettah_average' is numeric
    df['pettah_average'] = pd.to_numeric(df['pettah_average'], errors='coerce')

    # Transform 'pettah_average' by dividing 4-digit values by 50
    df['pettah_average'] = df['pettah_average'].apply(
        lambda x: x / 50 if pd.notnull(x) and len(str(int(x))) == 4 else x
    )

    # Save the DataFrame to an .xlsx file
    df.to_excel(config.local_data_file , index=False)

    return df