"""Supervised learning model training."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime
import warnings

import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso,
    LogisticRegression
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

# Optional: XGBoost and LightGBM
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger("ml_portfolio")


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str
) -> BaseEstimator:
    """
    Train a baseline model for comparison.

    Args:
        X_train: Training features
        y_train: Training target
        task_type: 'regression' or 'classification'

    Returns:
        Trained baseline model

    Note:
        Baseline uses mean prediction for regression and mode for classification.
        This establishes the minimum performance any model should beat.
    """
    if task_type == "regression":
        model = DummyRegressor(strategy="mean")
    else:
        model = DummyClassifier(strategy="most_frequent")

    model.fit(X_train, y_train)

    # Cross-validation score
    scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)

    logger.info(
        f"Baseline model trained. CV score: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    return model


def train_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    cv: int = 5
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a linear model with regularization.

    Args:
        X_train: Training features
        y_train: Training target
        task_type: 'regression' or 'classification'
        cv: Number of cross-validation folds

    Returns:
        Tuple of (trained model, training info dict)
    """
    if task_type == "regression":
        model = Ridge(random_state=42)
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        scoring = 'neg_mean_squared_error'
    else:
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2']
        }
        scoring = 'accuracy'

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    training_info = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
    }

    logger.info(
        f"Linear model trained. Best params: {grid_search.best_params_}, "
        f"Best score: {grid_search.best_score_:.4f}"
    )

    return grid_search.best_estimator_, training_info


def train_tree_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    model_type: str = "random_forest",
    cv: int = 5,
    random_state: int = 42
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a tree-based ensemble model with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target
        task_type: 'regression' or 'classification'
        model_type: 'random_forest', 'xgboost', or 'lightgbm'
        cv: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Tuple of (trained model, training info dict)

    Design Decision:
        Using RandomizedSearchCV for larger hyperparameter spaces to balance
        exploration with computational efficiency. n_iter=15 provides good
        coverage while keeping training time reasonable.
    """
    if model_type == "random_forest":
        if task_type == "regression":
            model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        else:
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1)

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    elif model_type == "xgboost" and XGBOOST_AVAILABLE:
        if task_type == "regression":
            model = XGBRegressor(
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            model = XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric='logloss'
            )

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        if task_type == "regression":
            model = LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
        else:
            model = LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0]
        }

    else:
        raise ValueError(f"Unknown or unavailable model type: {model_type}")

    scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'

    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=15,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random_search.fit(X_train, y_train)

    training_info = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
    }

    logger.info(
        f"{model_type} trained. Best params: {random_search.best_params_}, "
        f"Best score: {random_search.best_score_:.4f}"
    )

    return random_search.best_estimator_, training_info


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    cv: int = 5,
    random_state: int = 42,
    max_samples: int = 5000
) -> Tuple[Optional[BaseEstimator], Dict[str, Any]]:
    """
    Train an SVM model.

    Args:
        X_train: Training features
        y_train: Training target
        task_type: 'regression' or 'classification'
        cv: Number of cross-validation folds
        random_state: Random seed
        max_samples: Maximum samples for training (SVM is slow on large data)

    Returns:
        Tuple of (trained model or None, training info dict)

    Note:
        SVM is computationally expensive. For datasets larger than max_samples,
        we subsample to keep training time manageable.
    """
    training_info = {}

    # Check dataset size and subsample if needed
    if len(X_train) > max_samples:
        logger.warning(
            f"Dataset too large for SVM ({len(X_train)} > {max_samples}). "
            f"Subsampling to {max_samples} samples."
        )
        indices = np.random.RandomState(random_state).choice(
            len(X_train), max_samples, replace=False
        )
        X_sub = X_train[indices]
        y_sub = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        training_info['subsampled'] = True
        training_info['original_size'] = len(X_train)
    else:
        X_sub = X_train
        y_sub = y_train

    if task_type == "regression":
        model = SVR()
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        scoring = 'neg_mean_squared_error'
    else:
        model = SVC(random_state=random_state, probability=True)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        scoring = 'accuracy'

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_sub, y_sub)

    training_info.update({
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
    })

    logger.info(
        f"SVM trained. Best params: {grid_search.best_params_}, "
        f"Best score: {grid_search.best_score_:.4f}"
    )

    return grid_search.best_estimator_, training_info


def save_model(
    model: BaseEstimator,
    filepath: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save a trained model to disk with metadata.

    Args:
        model: Trained model to save
        filepath: Path to save the model
        metadata: Optional metadata to save alongside model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Handle versioning — rename existing file
    if filepath.exists():
        version = 1
        while True:
            versioned_path = filepath.with_suffix(f'.v{version}.joblib')
            if not versioned_path.exists():
                filepath.rename(versioned_path)
                break
            version += 1

    # Prepare save data
    final_metadata = (metadata or {}).copy()
    final_metadata.update({
        'saved_at': datetime.now().isoformat(),
        'model_type': type(model).__name__
    })

    save_data = {
        'model': model,
        'metadata': final_metadata
    }

    joblib.dump(save_data, filepath)
    logger.info(f"Saved model to {filepath}")


def load_model(filepath: Path) -> Tuple[BaseEstimator, Dict]:
    """
    Load a saved model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Tuple of (model, metadata)
    """
    save_data = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")
    return save_data['model'], save_data.get('metadata', {})


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    cv: int = 5,
    random_state: int = 42,
    experiment_name: str = "house_prices_experiment"
) -> Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]:
    """
    Train and tune all available models with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        task_type: 'regression' or 'classification'
        cv: Number of cross-validation folds
        random_state: Random seed
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary mapping model names to (model, info) tuples
    """
    try:
        import mlflow
        import mlflow.sklearn
        HAS_MLFLOW = True
        mlflow.set_experiment(experiment_name)
    except ImportError:
        HAS_MLFLOW = False
        logger.warning("MLflow not found. Skipping experiment tracking.")

    results = {}

    # Helper to log to MLflow safely
    def log_run(run_name, model, info, metrics=None):
        if not HAS_MLFLOW:
            return
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", run_name)
            
            # Log params
            if "best_params" in info:
                mlflow.log_params(info["best_params"])
            
            # Log metrics
            if "best_score" in info:
                mlflow.log_metric("cv_best_score", info["best_score"])
            if "cv_mean_score" in info: # Baseline/manual
                mlflow.log_metric("cv_mean_score", info["cv_mean_score"])
            if metrics:
                mlflow.log_metrics(metrics)
                
            # Log model
            if model:
                mlflow.sklearn.log_model(model, "model")

    # 1. Baseline
    logger.info("=" * 50)
    logger.info("Training Baseline...")
    baseline_model = train_baseline_model(X_train, y_train, task_type)
    
    # Calculate a simple metric for baseline for logging
    baseline_score = 0
    if hasattr(baseline_model, "predict"):
        try:
            preds = baseline_model.predict(X_train)
            if task_type == "regression":
                baseline_score = -((y_train - preds) ** 2).mean() # Negative MSE
            else:
                baseline_score = (y_train == preds).mean() # Accuracy
        except Exception:
            pass
            
    results["Baseline"] = (baseline_model, {"cv_mean_score": baseline_score})
    log_run("Baseline", baseline_model, results["Baseline"][1])

    # 2. Linear Model
    logger.info("=" * 50)
    logger.info("Training Linear model...")
    linear_model, linear_info = train_linear_model(
        X_train, y_train, task_type, cv=cv
    )
    results["Linear"] = (linear_model, linear_info)
    log_run("Linear", linear_model, linear_info)

    # 3. Tree Ensembles
    tree_models = [
        ("RandomForest", "random_forest"),
    ]
    if XGBOOST_AVAILABLE:
        tree_models.append(("XGBoost", "xgboost"))
    if LIGHTGBM_AVAILABLE:
        tree_models.append(("LightGBM", "lightgbm"))

    for model_name, model_type in tree_models:
        logger.info("=" * 50)
        logger.info(f"Training {model_name}...")
        model, info = train_tree_ensemble(
            X_train, y_train, task_type, model_type=model_type, cv=cv, random_state=random_state
        )
        results[model_name] = (model, info)
        log_run(model_name, model, info)

    # 4. SVM
    if len(X_train) <= 5000: # Reduced limit for speed in this demo, strict limit
        logger.info("=" * 50)
        logger.info("Training SVM...")
        try:
            svm_model, svm_info = train_svm(
                X_train, y_train, task_type, cv=cv, random_state=random_state
            )
            if svm_model:
                results["SVM"] = (svm_model, svm_info)
                log_run("SVM", svm_model, svm_info)
            else:
                results["SVM"] = (None, {})
        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            results["SVM"] = (None, {})
    else:
        logger.warning(f"Skipping SVM due to dataset size ({len(X_train)} > 5000)")
        results["SVM"] = (None, {})

    logger.info("=" * 50)
    logger.info(f"Finished training {len(results)} models")
    
    return results
