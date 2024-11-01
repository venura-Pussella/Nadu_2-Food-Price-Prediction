import pandas as pd
from box import ConfigBox 

def data_read_clean_missing_values(config):

    # Read the Excel file into a DataFrame
    df = pd.read_excel(config.unzip_data_dir)
    
    # Replace missing values in 'items' with 'Rice (Rs/kg)_Nadu 2'
    df['items'] = df['items'].fillna('Rice (Rs/kg)_Nadu 2')
    
    # Interpolate missing values in 'pettah_average' using linear interpolation
    df['pettah_average'] = df['pettah_average'].interpolate(method='linear', limit_direction='both')
    
    return df

def drop_unnecessary_columns(df):

    columns_to_drop = [
        'items', 'pettah_min_value', 'pettah_max_value', 
        'food_inflation_Base_2013', 'percipitation', 
        'Bankrupt', 'pettah_range', 'pettah_midpoint'
    ]
    
    # Drop the specified columns
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df


def save_preprocessed_excel(config: ConfigBox, df: pd.DataFrame):
    # Save the DataFrame to the specified file path as an Excel file
    df.to_excel(config.local_data_file, index=False)
    print(f"File saved at: {config.local_data_file}")