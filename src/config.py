"""Centralized configuration management for ML pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import os


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    kaggle_id: str
    target_column: str
    task_type: str  # 'regression' or 'classification'
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    drop_features: List[str] = field(default_factory=list)
    

@dataclass
class ModelConfig:
    """Configuration for model training."""
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    

@dataclass 
class HyperparameterConfig:
    """Hyperparameter grids for different models."""
    
    random_forest: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    xgboost: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    })
    
    lightgbm: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'num_leaves': [31, 50, 100]
    })
    
    svm: Dict[str, List[Any]] = field(default_factory=lambda: {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    })


@dataclass
class ProjectConfig:
    """Main project configuration."""
    
    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    figures_dir: Path = Path("figures")
    reports_dir: Path = Path("reports")
    
    # Dataset
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="house-prices",
        kaggle_id="house-prices-advanced-regression-techniques",
        target_column="SalePrice",
        task_type="regression"
    ))
    
    # Model settings
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Note: robust implementation would allow partial updates
        # For now, simplistic loading
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Default configuration instance
CONFIG = ProjectConfig()


# Supported datasets registry
DATASETS = {
    'house-prices': DatasetConfig(
        name='house-prices',
        kaggle_id='house-prices-advanced-regression-techniques',
        target_column='SalePrice',
        task_type='regression',
        categorical_features=[
            'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
            'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
            'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
            'SaleCondition'
        ],
        numerical_features=[
            'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
            'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
            'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
            'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal', 'MoSold', 'YrSold'
        ],
        drop_features=['Id', 'Order', 'PID']
    ),
    'titanic': DatasetConfig(
        name='titanic',
        kaggle_id='titanic',
        target_column='Survived',
        task_type='classification',
        categorical_features=['Sex', 'Embarked', 'Pclass'],
        numerical_features=['Age', 'Fare', 'SibSp', 'Parch'],
        drop_features=['PassengerId', 'Name', 'Ticket', 'Cabin']
    ),
    'credit-card-fraud': DatasetConfig(
        name='credit-card-fraud',
        kaggle_id='creditcardfraud',
        target_column='Class',
        task_type='classification',
        numerical_features=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'],
        categorical_features=[],
        drop_features=[]
    ),
    'bank-marketing': DatasetConfig(
        name='bank-marketing',
        kaggle_id='bank-marketing',
        target_column='y',
        task_type='classification',
        categorical_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'],
        numerical_features=['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
        drop_features=[]
    )
}
