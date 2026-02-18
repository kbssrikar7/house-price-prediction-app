"""Data acquisition and loading functionality."""

import os
import zipfile
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ml_portfolio")

# Dataset configurations
DATASET_CONFIGS = {
    "house-prices-advanced-regression-techniques": {
        "kaggle_path": "competitions/house-prices-advanced-regression-techniques",
        "target_column": "SalePrice",
        "task_type": "regression",
        "train_file": "train.csv",
        "alt_files": ["AmesHousing.csv", "ames-housing-dataset.csv"],
        "test_file": "test.csv",
    },
    "titanic": {
        "kaggle_path": "competitions/titanic",
        "target_column": "Survived",
        "task_type": "classification",
        "train_file": "train.csv",
        "test_file": "test.csv",
    },
    "creditcardfraud": {
        "kaggle_path": "mlg-ulb/creditcardfraud",
        "target_column": "Class",
        "task_type": "classification",
        "train_file": "creditcard.csv",
        "test_file": None,
    },
    "bank-marketing": {
        "kaggle_path": "henriqueyamahata/bank-marketing",
        "target_column": "y",
        "task_type": "classification",
        "train_file": "bank-additional-full.csv",
        "test_file": None,
    },
}


def download_dataset(
    dataset_name: str = "house-prices-advanced-regression-techniques",
    output_dir: str = "data/raw/"
) -> Path:
    """
    Download a dataset from Kaggle API.

    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to store downloaded files

    Returns:
        Path to the downloaded data directory

    Raises:
        ValueError: If dataset_name is not supported
        RuntimeError: If Kaggle credentials are not configured
    """
    if dataset_name not in DATASET_CONFIGS:
        supported = list(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from: {supported}"
        )

    config = DATASET_CONFIGS[dataset_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    train_file = config.get("train_file", "train.csv")
    existing_file = output_path / train_file
    if existing_file.exists():
        logger.info(f"Dataset already exists at {existing_file}")
        return output_path

    # Check for Kaggle credentials
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        raise RuntimeError(
            "Kaggle credentials not found. Please place kaggle.json in "
            f"{kaggle_config.parent}. See: https://www.kaggle.com/docs/api\n"
            "Alternatively, manually download the dataset CSV and place it in "
            f"{output_path}"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        kaggle_path = config["kaggle_path"]

        if kaggle_path.startswith("competitions/"):
            # Competition dataset
            competition_name = kaggle_path.replace("competitions/", "")
            api.competition_download_files(
                competition_name,
                path=str(output_path)
            )
        else:
            # Regular dataset
            api.dataset_download_files(
                kaggle_path,
                path=str(output_path),
                unzip=True
            )

        # Unzip if needed
        for zip_file in output_path.glob("*.zip"):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            zip_file.unlink()  # Remove zip after extraction

        logger.info(f"Downloaded dataset '{dataset_name}' to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def load_data(
    data_path: Path,
    dataset_name: str = "house-prices-advanced-regression-techniques"
) -> pd.DataFrame:
    """
    Load dataset from disk into a pandas DataFrame.

    Args:
        data_path: Path to the data directory
        dataset_name: Name of the dataset for configuration lookup

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If loaded data is empty or invalid
    """
    config = DATASET_CONFIGS.get(dataset_name, {})
    train_file = config.get("train_file", "train.csv")

    file_path = Path(data_path) / train_file

    if not file_path.exists():
        # Try alternate filenames from config
        alt_files = config.get("alt_files", [])
        for alt in alt_files:
            alt_path = Path(data_path) / alt
            if alt_path.exists():
                file_path = alt_path
                logger.info(f"Using alternate file: {file_path}")
                break
        else:
            # Try finding any CSV file
            csv_files = list(Path(data_path).glob("*.csv"))
            if csv_files:
                file_path = csv_files[0]
                logger.warning(f"Using found CSV file: {file_path}")
            else:
                raise FileNotFoundError(f"No CSV file found in {data_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Normalize column names (handle spaces, special chars)
    df = normalize_column_names(df)

    # Validation
    validation_result = validate_dataset(df)
    if not validation_result["is_valid"]:
        raise ValueError(f"Invalid dataset: {validation_result['errors']}")

    logger.info(
        f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns"
    )
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that a DataFrame meets minimum requirements.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []

    # Check for minimum rows
    if len(df) < 100:
        errors.append(f"Dataset has only {len(df)} rows, minimum 100 required")

    # Check for minimum columns
    if len(df.columns) < 2:
        errors.append(f"Dataset has only {len(df.columns)} columns, minimum 2 required")

    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        warnings.append(f"Columns with all null values: {null_cols}")

    # Check for high missing value percentages
    high_missing = []
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 80:
            high_missing.append(f"{col}: {missing_pct:.1f}%")
    if high_missing:
        warnings.append(f"Columns with >80% missing: {high_missing}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: remove spaces, special characters.

    Handles the AmesHousing.csv format (spaces) vs competition format (no spaces).
    E.g. 'Overall Qual' -> 'OverallQual', '1st Flr SF' -> '1stFlrSF'
    """
    rename_map = {}
    for col in df.columns:
        new_col = col.replace(' ', '')
        if new_col != col:
            rename_map[col] = new_col

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Normalized {len(rename_map)} column names (removed spaces)")

    # Drop 'Order' or 'PID' columns if present (Ames dataset specific)
    drop_cols = [c for c in ['Order', 'PID'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped index columns: {drop_cols}")

    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split (for classification)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Determine stratify parameter
    stratify_col = y if stratify and y.nunique() < 50 else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Second split: train vs val
    val_proportion = val_size / (1 - test_size)
    stratify_temp = y_temp if stratify and y.nunique() < 50 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_proportion,
        random_state=random_state,
        stratify=stratify_temp
    )

    logger.info(
        f"Split data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Configuration dictionary
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name].copy()
