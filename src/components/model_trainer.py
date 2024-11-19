from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional , Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np


def sequence_creation_with_forecast(config):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(config.local_data_path)

    # Ensure 'date' column is present and set as the index
    if 'date' not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Ensure the dataset has sufficient columns and rows
    if df.shape[1] < 1:
        raise ValueError("The dataset must contain at least one feature column.")

    # Define sequence length and forecast horizon
    sequence_length = config.sequence_length  # e.g., 30
    forecast_horizon = config.forecast_horizon  # e.g., 5

    # Limit data to the latest 1825 data points
    data = df[-1825:].values
    dates = df.index[-1825:]

    # Create sequences and labels
    sequences, labels, training_dates, forecast_dates = [], [], [], []
    for i in range(len(data) - sequence_length):
        # Ensure we don't go out of bounds for forecast horizon
        if i + sequence_length + forecast_horizon > len(data):
            break

        seq = data[i:i + sequence_length]
        label = data[i + sequence_length:i + sequence_length + forecast_horizon, 0]  # Assuming the target is the first column

        sequences.append(seq)
        labels.append(label)
        training_dates.append(dates[i:i + sequence_length])
        forecast_dates.append(dates[i + sequence_length:i + sequence_length + forecast_horizon])

    sequences = np.array(sequences)
    labels = np.array(labels)

    # Separate training and testing data
    train_x = sequences[:-1]
    train_y = labels[:-1]
    train_dates = training_dates[:-1]

    test_x = sequences[-1:]
    test_y = labels[-1:]
    test_dates = forecast_dates[-1] if len(forecast_dates) > 0 else dates[-forecast_horizon:]

    return train_x, test_x, train_y, test_y, train_dates, test_dates

# def sequence_creation_with_forecast(config):
#     # Read the Excel file into a DataFrame
#     df = pd.read_excel(config.local_data_path)

#     # Ensure 'date' column is present and set as the index
#     if 'date' not in df.columns:
#         raise ValueError("The dataset must contain a 'date' column.")
    
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)

#     # Ensure the dataset has sufficient columns and rows
#     if df.shape[1] < 1:
#         raise ValueError("The dataset must contain at least one feature column.")

#     # Define sequence length and forecast horizon
#     sequence_length = config.sequence_length  # e.g., 30
#     forecast_horizon = config.forecast_horizon  # e.g., 5

#     # Limit data to the latest 1825 data points
#     data = df[-1825:].values
#     dates = df.index[-1825:]

#     # Create sequences and labels
#     sequences, labels, training_dates, forecast_dates = [], [], [], []
#     for i in range(len(data) - sequence_length - forecast_horizon):
#         seq = data[i:i + sequence_length]

#         label = data[i + sequence_length:i + sequence_length + forecast_horizon, 0]  # Assuming the target is the first column
        
#         sequences.append(seq)
#         labels.append(label)
#         training_dates.append(dates[i:i + sequence_length])
#         forecast_dates.append(dates[i + sequence_length:i + sequence_length + forecast_horizon])

#     sequences = np.array(sequences)
#     labels = np.array(labels)

#     # Separate training and testing data
#     train_x = sequences[:-1]
#     train_y = labels[:-1]
#     train_dates = training_dates[:-1]

#     test_x = sequences[-1:]
#     test_y = labels[-1:]
#     test_dates = forecast_dates[-1]

#     # print("Train X shape:", train_x.shape)  # Expected: (samples, 30, num_features)
#     # print("Train Y shape:", train_y.shape)  # Expected: (samples, 5)
#     # print("Test X shape:", test_x.shape)    # Expected: (1, 30, num_features)
#     # print("Test Y shape:", test_y.shape)    # Expected: (1, 5)
#     # print("Example forecast dates for the first test sample:", test_dates[0])
#     # print("Example training dates for the first test sample:", train_dates[0])

#     return train_x, test_x, train_y, test_y, train_dates, test_dates


def save_train_test_data_to_excel(train_x, test_x, train_y, test_y, train_dates, test_dates, config):
    """
    Save training and testing data (features, labels, and dates) to separate Excel files.
    """
    # Flatten 3D arrays into 2D for saving
    train_x_df = pd.DataFrame(train_x.reshape(train_x.shape[0], -1))  
    test_x_df = pd.DataFrame(test_x.reshape(test_x.shape[0], -1))    

    # Save labels and dates
    train_y_df = pd.DataFrame(train_y, columns=[f"day_{i+1}" for i in range(train_y.shape[1])])
    test_y_df = pd.DataFrame(test_y, columns=[f"day_{i+1}" for i in range(test_y.shape[1])])

    train_dates_df = pd.DataFrame([list(d) for d in train_dates])
    test_dates_df = pd.DataFrame([list(test_dates)])

    # Write DataFrames to Excel files
    train_x_df.to_excel(config.train_x_data_file, index=False)
    train_y_df.to_excel(config.train_y_data_file, index=False)
    train_dates_df.to_excel(config.train_dates_data_file, index=False)

    test_x_df.to_excel(config.test_x_data_file, index=False)
    test_y_df.to_excel(config.test_y_data_file, index=False)
    test_dates_df.to_excel(config.test_dates_data_file, index=False)

    print(f"Training and testing data written to respective files successfully.")

def lstm_model_trainer(train_x, train_y, config):

    # Create the LSTM model using the provided config
    model = Sequential()

    model.add(Conv1D(filters=config.filters, kernel_size=config.kernel_size, activation=config.activation1, input_shape=(train_x.shape[1], train_x.shape[2])))

    # Add Bidirectional LSTM layers with dropout
    model.add(Bidirectional(LSTM(units=config.n_units_layer1, return_sequences=True)))
    model.add(Dropout(config.dropout_rate))

    model.add(Bidirectional(LSTM(units=config.n_units_layer2, return_sequences=True)))
    model.add(Dropout(config.dropout_rate))

    model.add(Bidirectional(LSTM(units=config.n_units_layer3, return_sequences=False)))
    model.add(Dropout(config.dropout_rate))

    # Add a dense output layer
    model.add(Dense(units=config.n_units_layer4,activation=config.activation2))

    # Compile the model using the config
    model.compile(optimizer=config.optimizer, loss=config.loss_function)

    # Create a checkpoint callback for saving the best model
    model_checkpoint = ModelCheckpoint(filepath=str(config.model_checkpoint_path), 
                                       monitor='val_loss', 
                                       save_best_only=True)
    # Train the model
    history = model.fit(
        train_x, train_y,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=[model_checkpoint]
    )

    return model, history