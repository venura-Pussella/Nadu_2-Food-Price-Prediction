import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from box import ConfigBox

def box_cox_transformation(config):

    # Read the Excel file into a DataFrame
    df = pd.read_excel(config.data_path)

    # Apply Box-Cox Transformation to 'pettah_average'
    df['pettah_average'], lambda_value = boxcox(df['pettah_average'].replace(0, 0.01))
    
    return df , lambda_value

def min_max_scale(df):

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)

    # Create a DataFrame from the scaled data with the same column names
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    return scaled_df

def remove_zeros_in_df(df):

    # Identify rows that have any '0' values
    rows_with_zeros = (df == 0).any(axis=1)

    # Filter out those rows (keep only rows without '0' values)
    cleaned_df = df[~rows_with_zeros]

    return cleaned_df

def save_preprocessed_excel(config: ConfigBox, df: pd.DataFrame):
    # Save the DataFrame to the specified file path as an Excel file
    df.to_excel(config.local_data_file, index=False)
    print(f"File saved at: {config.local_data_file}")