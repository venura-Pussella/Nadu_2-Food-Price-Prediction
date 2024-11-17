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

    # Extract values and date index
    scaled_data = df.values  # Convert DataFrame to numpy array
    date_index = df.index  # Extract the date index

    # Extract configuration parameters
    sequence_length = config.sequence_length
    forecast_horizon = config.forecast_horizon

    # Initialize lists to store sequences, labels, and forecast dates
    sequences, labels, forecast_dates = [], [], []

    for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
        # Input sequence for the specified sequence length
        seq = scaled_data[i:i + sequence_length]

        # Labels: Next 'forecast_horizon' days (using the first feature for labels)
        label = scaled_data[i + sequence_length:i + sequence_length + forecast_horizon, 0]

        # Corresponding dates for the forecast horizon
        forecast_date_range = date_index[i + sequence_length:i + sequence_length + forecast_horizon]

        # Append to lists
        sequences.append(seq)
        labels.append(label)
        forecast_dates.append(forecast_date_range)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]
    train_dates, test_dates = forecast_dates[:train_size], forecast_dates[train_size:]

    return train_x, test_x, train_y, test_y, train_dates, test_dates


def save_train_test_data_to_excel(train_x, test_x, train_y, test_y, config):

    # Convert train_x, train_y, test_x, test_y to DataFrames
    train_x_df = pd.DataFrame(train_x.reshape(train_x.shape[0], -1))  # Flatten 3D array to 2D
    train_y_df = pd.DataFrame(train_y, columns=[config.target_column])

    test_x_df = pd.DataFrame(test_x.reshape(test_x.shape[0], -1))  # Flatten 3D array to 2D
    test_y_df = pd.DataFrame(test_y, columns=[config.target_column])

    # Save DataFrames to Excel files defined in config
    train_x_path = config.train_x_data_file
    train_y_path = config.train_y_data_file
    test_x_path = config.test_x_data_file
    test_y_path = config.test_y_data_file

    train_x_df.to_excel(train_x_path, index=False)
    train_y_df.to_excel(train_y_path, index=False)

    test_x_df.to_excel(test_x_path, index=False)
    test_y_df.to_excel(test_y_path, index=False)

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