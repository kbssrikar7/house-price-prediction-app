"""Model evaluation and comparison functionality."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger("ml_portfolio")


def evaluate_regression_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a regression model on test data.

    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics: RMSE, MAE, R2, MAPE
    """
    y_pred = model.predict(X_test)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
    }

    # MAPE (handle division by zero)
    nonzero_mask = y_test != 0
    if nonzero_mask.any():
        metrics['MAPE'] = np.mean(
            np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])
        ) * 100
    else:
        metrics['MAPE'] = np.nan

    logger.info(f"Regression metrics: RMSE={metrics['RMSE']:.4f}, "
                f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    return metrics


def evaluate_classification_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a classification model on test data.

    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics: Accuracy, Precision, Recall, F1, ROC_AUC, PR_AUC
    """
    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # ROC-AUC and PR-AUC (require probability predictions)
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)

            if y_proba.shape[1] == 2:
                # Binary classification
                metrics['ROC_AUC'] = roc_auc_score(y_test, y_proba[:, 1])
                metrics['PR_AUC'] = average_precision_score(y_test, y_proba[:, 1])
            else:
                # Multiclass
                metrics['ROC_AUC'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='weighted'
                )
                metrics['PR_AUC'] = np.nan
        except Exception as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
            metrics['ROC_AUC'] = np.nan
            metrics['PR_AUC'] = np.nan
    else:
        metrics['ROC_AUC'] = np.nan
        metrics['PR_AUC'] = np.nan

    logger.info(f"Classification metrics: Accuracy={metrics['Accuracy']:.4f}, "
                f"F1={metrics['F1']:.4f}")

    return metrics


def compare_models(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary mapping model names to trained models
        X_test: Test features
        y_test: Test target
        task_type: 'regression' or 'classification'

    Returns:
        DataFrame with comparison metrics for all models
    """
    results = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        if task_type == "regression":
            metrics = evaluate_regression_model(model, X_test, y_test)
        else:
            metrics = evaluate_classification_model(model, X_test, y_test)

        metrics['Model'] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Model')

    # Sort by primary metric
    if task_type == "regression":
        comparison_df = comparison_df.sort_values('RMSE')
    else:
        comparison_df = comparison_df.sort_values('F1', ascending=False)

    return comparison_df


def generate_evaluation_report(
    comparison_df: pd.DataFrame,
    output_path: Path,
    task_type: str
) -> None:
    """
    Generate and save evaluation report in CSV and Markdown formats.

    Args:
        comparison_df: Model comparison DataFrame
        output_path: Directory to save reports
        task_type: 'regression' or 'classification'
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_path / "model_comparison.csv"
    comparison_df.to_csv(csv_path)

    # Generate Markdown report
    md_path = output_path / "model_comparison.md"

    with open(md_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Task Type:** {task_type.title()}\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(comparison_df.to_markdown())
        f.write("\n\n")

        # Best model
        if task_type == "regression":
            best_model = comparison_df['RMSE'].idxmin()
            f.write(f"## Best Model: {best_model}\n\n")
            f.write(f"**RMSE:** {comparison_df.loc[best_model, 'RMSE']:.4f}\n\n")
            f.write(f"**R²:** {comparison_df.loc[best_model, 'R2']:.4f}\n\n")
        else:
            best_model = comparison_df['F1'].idxmax()
            f.write(f"## Best Model: {best_model}\n\n")
            f.write(f"**F1 Score:** {comparison_df.loc[best_model, 'F1']:.4f}\n\n")

    logger.info(f"Saved evaluation report to {output_path}")


def get_best_model(
    models: Dict[str, BaseEstimator],
    comparison_df: pd.DataFrame,
    task_type: str
) -> Tuple[str, BaseEstimator]:
    """
    Get the best performing model based on comparison results.

    Args:
        models: Dictionary of trained models
        comparison_df: Comparison DataFrame
        task_type: 'regression' or 'classification'

    Returns:
        Tuple of (model_name, model)
    """
    if task_type == "regression":
        best_name = comparison_df['RMSE'].idxmin()
    else:
        best_name = comparison_df['F1'].idxmax()

    return best_name, models[best_name]
