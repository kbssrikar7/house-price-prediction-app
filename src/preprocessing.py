"""Feature engineering and preprocessing functionality."""

import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer

logger = logging.getLogger("ml_portfolio")


def engineer_features(
    df: pd.DataFrame,
    dataset_type: str = "house-prices"
) -> pd.DataFrame:
    """
    Create engineered features based on dataset type.

    Args:
        df: Input DataFrame
        dataset_type: Type of dataset for domain-specific features

    Returns:
        DataFrame with engineered features

    Note:
        Feature engineering rationale:
        - TotalSF: Combined square footage is a strong price predictor
        - TotalBath: Bathrooms consolidated for interpretability
        - HouseAge: Age affects condition and style preferences
        - IsNew: Binary indicator for new constructions premium
        - QualityArea: Interaction between quality and living area
    """
    df = df.copy()
    original_cols = len(df.columns)

    if dataset_type == "house-prices":
        # Total square footage (strong predictor for price)
        sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in df.columns for col in sf_cols):
            df['TotalSF'] = df[sf_cols].sum(axis=1)

        # Total bathrooms
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in df.columns for col in bath_cols):
            df['TotalBath'] = (
                df['FullBath'] + 0.5 * df['HalfBath'] +
                df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
            )

        # House age
        if 'YearBuilt' in df.columns:
            current_year = 2026
            df['HouseAge'] = current_year - df['YearBuilt']
            df['IsNew'] = (df['HouseAge'] <= 5).astype(int)

        # Remodel age
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            df['RemodAge'] = 2026 - df['YearRemodAdd']
            df['WasRemodeled'] = (
                df['YearRemodAdd'] != df['YearBuilt']
            ).astype(int)

        # Quality * Area interactions
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['QualityArea'] = df['OverallQual'] * df['GrLivArea']

        new_cols = len(df.columns) - original_cols
        logger.info(f"Engineered {new_cols} new features for house-prices dataset")

    elif dataset_type == "titanic":
        # Family size
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Title extraction from name
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
            rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don',
                          'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
            df['Title'] = df['Title'].replace(rare_titles, 'Rare')
            df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')

    elif dataset_type == "creditcardfraud":
        # For credit card fraud, features are already anonymized (V1-V28)
        if 'Amount' in df.columns:
            df['Amount_Log'] = np.log1p(df['Amount'])
            df['Amount_Bin'] = pd.qcut(
                df['Amount'], q=10, labels=False, duplicates='drop'
            )

    return df


def handle_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "most_frequent"
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame
        numerical_strategy: Strategy for numerical columns (mean, median, constant)
        categorical_strategy: Strategy for categorical columns (most_frequent, constant)

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numerical columns
    if numerical_cols:
        num_imputer = SimpleImputer(strategy=numerical_strategy)
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    logger.info(f"Imputed missing values: {numerical_strategy} for numerical, "
                f"{categorical_strategy} for categorical")

    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.

    Args:
        df: Input DataFrame
        columns: Columns to check (None for all numerical)
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with boolean outlier indicators
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.DataFrame(index=df.index)

    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask[col] = z_scores > threshold
            else:
                outlier_mask[col] = False

    return outlier_mask


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    numerical_strategy: str = "standard",
    categorical_strategy: str = "onehot"
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline using ColumnTransformer.

    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        numerical_strategy: Scaling strategy ('standard', 'minmax', 'robust')
        categorical_strategy: Encoding strategy ('onehot', 'ordinal')

    Returns:
        Configured ColumnTransformer pipeline

    Design Decision:
        Using ColumnTransformer allows parallel processing of different
        feature types while maintaining a single pipeline interface.
        StandardScaler is default as it handles outliers better than MinMax.
    """
    # Numerical pipeline
    if numerical_strategy == "standard":
        scaler = StandardScaler()
    elif numerical_strategy == "minmax":
        scaler = MinMaxScaler()
    elif numerical_strategy == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown numerical strategy: {numerical_strategy}")

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])

    # Categorical pipeline
    if categorical_strategy == "onehot":
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif categorical_strategy == "ordinal":
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1)
    else:
        raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder)
    ])

    # Combine pipelines
    transformers = []
    if numerical_features:
        transformers.append(('numerical', numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(('categorical', categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )

    logger.info(
        f"Built preprocessing pipeline: {len(numerical_features)} numerical, "
        f"{len(categorical_features)} categorical features"
    )

    return preprocessor


def get_feature_lists(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical features.

    Args:
        df: Input DataFrame
        target_column: Name of target column to exclude
        drop_columns: Additional columns to exclude

    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    drop_columns = list(drop_columns or [])
    drop_columns.append(target_column)

    # Get feature columns
    feature_df = df.drop(columns=drop_columns, errors='ignore')

    numerical_features = feature_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_features = feature_df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    logger.info(
        f"Identified {len(numerical_features)} numerical and "
        f"{len(categorical_features)} categorical features"
    )

    return numerical_features, categorical_features
