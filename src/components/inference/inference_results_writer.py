import pandas as pd
from azure.cosmos import CosmosClient , exceptions
from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name

# Initialize the Cosmos client
client = CosmosClient(cosmos_url, cosmos_key)
# Get the database and container clients
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Function to fetch Pettah Average for a given date from Cosmos DB
def fetch_pettah_average(date):
    """
    Query Cosmos DB to fetch the Pettah Average for a given date.
    """
    query = f"""

    SELECT c.date, c.pettah_average
    FROM c
    WHERE c.item = "Nadu 2" AND STARTSWITH(c.date, "{date}")

    """
    try:
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        if items:
            return items[0]['pettah_average']
        else:
            return None 
    except exceptions.CosmosHttpResponseError as e:
        print(f"An error occurred: {e.message}")
        return None    
    
# Function to update real values in the CSV
def update_real_values_in_csv(config):
    """
    Read a CSV, fetch Pettah Average values for dates, and save the updated CSV.
    """
    try:
        # Read the CSV
        df = pd.read_csv(config.data_path)
        
        # Ensure 'date' column is in the correct format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Apply the fetch_pettah_average function to the 'date' column
        df['real_value'] = df['date'].apply(fetch_pettah_average)

        # Save the updated DataFrame back to the same CSV
        df.to_csv(config.data_path, index=False)
        print(f"Updated CSV saved successfully at {config.data_path}")
    except Exception as e:
        print(f"An error occurred while updating the CSV: {e}")

# Example usage
if __name__ == '__main__':

    # Update real values in the CSV
    update_real_values_in_csv()