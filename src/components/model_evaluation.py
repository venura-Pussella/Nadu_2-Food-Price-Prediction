import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_evaluation(train_x, test_x, train_y, test_y,config):

    # Load the model
    best_model = tf.keras.models.load_model(config.model_path)

    # Predict on train data
    predicted_train_set_y = best_model.predict(train_x)

    # Predict on test data
    predicted_test_set_y = best_model.predict(test_x)

    # Calculate evaluation metrics for the training data
    train_r2 = r2_score(train_y, predicted_train_set_y)
    train_mae = mean_absolute_error(train_y, predicted_train_set_y)
    train_mse = mean_squared_error(train_y, predicted_train_set_y)
    train_rmse = np.sqrt(train_mse)

    # Calculate evaluation metrics for the test data
    test_r2 = r2_score(test_y, predicted_test_set_y)
    test_mae = mean_absolute_error(test_y, predicted_test_set_y)
    test_mse = mean_squared_error(test_y, predicted_test_set_y)
    test_rmse = np.sqrt(test_mse)

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data for saving
    metrics_data = {
        "Date": [current_time],  
        "Train R2": [train_r2],
        "Train MAE": [train_mae],
        "Train MSE": [train_mse],
        "Train RMSE": [train_rmse],
        "Test R2": [test_r2],
        "Test MAE": [test_mae],
        "Test MSE": [test_mse],
        "Test RMSE": [test_rmse]
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Save metrics to Excel (appending to the file if it exists)
    excel_file = config.metric_file_name
    try:
        # If the Excel file already exists, append the new data
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False, header=False, startrow=writer.sheets['Metrics'].max_row)
    except FileNotFoundError:
        # If the file doesn't exist, create a new file and write data
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)