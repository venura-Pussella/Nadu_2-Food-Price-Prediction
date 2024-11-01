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
    local_data_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    root_dir_train: Path
    root_dir_test: Path
    local_data_path: Path
    train_y_data_file: Path
    train_x_data_file: Path
    test_y_data_file: Path
    test_x_data_file: Path
    model_checkpoint_path: Path
    n_units_layer1: int
    n_units_layer2: int
    n_units_layer3: int
    activation: str
    dropout_rate: float
    sequence_length: int
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
