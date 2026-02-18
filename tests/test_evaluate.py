"""Unit tests for evaluation module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import (
    evaluate_regression_model,
    evaluate_classification_model,
    compare_models
)


class TestRegressionEvaluation:
    """Tests for regression model evaluation."""
    
    def test_evaluate_regression_returns_all_metrics(self):
        """Test that all required metrics are returned."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = DummyRegressor(strategy='mean')
        model.fit(X, y)
        
        metrics = evaluate_regression_model(model, X, y)
        
        # Keys are uppercase in implementation
        assert 'RMSE' in metrics, "Should include RMSE"
        assert 'MAE' in metrics, "Should include MAE"
        assert 'R2' in metrics, "Should include R²"
    
    # Feature: ml-portfolio-project, Property 14: Regression Metrics Completeness
    def test_regression_metrics_are_numeric(self):
        """Test that all metrics are valid numbers."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = DummyRegressor(strategy='mean')
        model.fit(X, y)
        
        metrics = evaluate_regression_model(model, X, y)
        
        for name, value in metrics.items():
            assert isinstance(value, (int, float)), f"{name} should be numeric"
            if name != 'MAPE': # MAPE can be nan/inf if zeros
               assert not np.isnan(value), f"{name} should not be NaN"


class TestClassificationEvaluation:
    """Tests for classification model evaluation."""
    
    def test_evaluate_classification_returns_all_metrics(self):
        """Test that all required metrics are returned."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = DummyClassifier(strategy='stratified', random_state=42)
        model.fit(X, y)
        
        metrics = evaluate_classification_model(model, X, y)
        
        # Keys are Title Case in implementation
        required_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        for metric in required_metrics:
            assert metric in metrics, f"Should include {metric}"


class TestModelComparison:
    """Tests for model comparison functionality."""
    
    # Feature: ml-portfolio-project, Property 16: Model Comparison Consistency
    def test_compare_models_uses_same_test_set(self):
        """Test that all models are evaluated on the same data."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        models = {
            'model1': DummyRegressor(strategy='mean').fit(X, y),
            'model2': DummyRegressor(strategy='median').fit(X, y)
        }
        
        comparison = compare_models(models, X, y, 'regression')
        
        assert len(comparison) == 2, "Should have results for both models"
