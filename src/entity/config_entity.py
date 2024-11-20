from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    unzip_data_dir: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    lambda_value: Path
    local_data_file: Path
    min_max_scaler_file:Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    root_dir_train: Path
    root_dir_test: Path
    local_data_path: Path
    train_y_data_file: Path
    train_x_data_file: Path
    full_train_x_data_file: Path
    full_train_y_data_file: Path
    full_train_x_dates_file: Path
    full_train_y_dates_file: Path
    test_y_data_file: Path
    test_x_data_file: Path
    train_x_dates_file: Path
    train_y_dates_file: Path
    test_x_dates_file: Path
    test_y_dates_file: Path
    model_checkpoint_path: Path
    model_checkpoint_path2: Path
    filters: int
    kernel_size: int
    activation1: str
    n_units_layer1: int
    n_units_layer2: int
    n_units_layer3: int
    n_units_layer4: int
    activation2: str
    dropout_rate: float
    sequence_length: int
    forecast_horizon: int
    optimizer: str
    loss_function: str
    epochs: int
    batch_size: int
    validation_split: float
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    metric_file_name: Path
