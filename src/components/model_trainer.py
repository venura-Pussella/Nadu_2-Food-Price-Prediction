from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

def sequence_creation_train_test_split(config):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(config.local_data_path)
 
    # Define sequence length and number of features
    sequence_length = config.sequence_length 
    num_features = len(df.columns)

    # Create sequences and corresponding labels
    sequences = []
    labels = []

    # Dynamically set target column based on config.target_column
    target_col_idx = df.columns.get_loc(config.target_column)  # Get index of target column

    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length, :]
        label = df.iloc[i+sequence_length, target_col_idx]  # since target is the first column
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(0.8 * len(sequences)) 
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]

    return train_x, test_x, train_y, test_y


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

    # Add Bidirectional LSTM layers with dropout
    model.add(Bidirectional(LSTM(units=config.n_units_layer1, return_sequences=True), 
                            input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(config.dropout_rate))

    model.add(Bidirectional(LSTM(units=config.n_units_layer2, return_sequences=True)))
    model.add(Dropout(config.dropout_rate))

    model.add(Bidirectional(LSTM(units=config.n_units_layer3, return_sequences=False)))
    model.add(Dropout(config.dropout_rate))

    # Add a dense output layer
    model.add(Dense(units=1,activation=config.activation))

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