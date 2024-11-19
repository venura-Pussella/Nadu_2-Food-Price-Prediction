import joblib
import pandas as pd
from box import ConfigBox
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

def box_cox_transformation(config):

    # Read the Excel file into a DataFrame
    df = pd.read_excel(config.data_path)

    # Apply Box-Cox Transformation to 'pettah_average'
    df['pettah_average'], lambda_value = boxcox(df['pettah_average'].replace(0, 0.01))

    # Write the lambda value to the specified file
    with open(config.lambda_value, 'w') as file:
        file.write(str(lambda_value))
    
    return df


def min_max_scale(df, config):
    
    # Ensure 'date' is not included in the scaling
    numeric_columns = df.drop(columns=['date'])  # Drop the 'date' column for scaling

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the numeric data
    scaled_data = scaler.fit_transform(numeric_columns)

    # Create a DataFrame for scaled data with the same column names
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns.columns, index=df.index)

    # Re-add the 'date' column to the scaled DataFrame
    scaled_df['date'] = df['date']

    # Reorder columns to place 'date' as the first column
    scaled_df = scaled_df[['date'] + [col for col in scaled_df.columns if col != 'date']]

    # Save the MinMaxScaler object to a .pkl file
    joblib.dump(scaler, config.min_max_scaler_file)

    return scaled_df

def remove_zeros_in_df(df):

    # Filter out rows where 'pettah_average' is 0
    cleaned_df = df[df['pettah_average'] != 0]

    return cleaned_df

def save_preprocessed_excel(config: ConfigBox, df: pd.DataFrame):
    # Save the DataFrame to the specified file path as an Excel file
    df.to_excel(config.local_data_file, index=False)
    print(f"File saved at: {config.local_data_file}")