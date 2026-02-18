"""Unit tests for supervised training module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_supervised import (
    train_baseline_model,
    train_linear_model,
    save_model,
    load_model
)


class TestBaselineModel:
    """Tests for baseline model training."""
    
    def test_baseline_regression_predicts_mean(self):
        """Test that baseline regression predicts the mean."""
        X = np.random.randn(100, 5)
        # Ensure y is numeric
        y = np.array([10.0] * 50 + [20.0] * 50)  # Mean = 15
        
        model = train_baseline_model(X, y, 'regression')
        predictions = model.predict(X)
        
        assert np.allclose(predictions, 15.0), "Should predict the mean"
    
    def test_baseline_classification_predicts_mode(self):
        """Test that baseline classification predicts the mode."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 30 + [1] * 70)  # Mode = 1
        
        model = train_baseline_model(X, y, 'classification')
        predictions = model.predict(X)
        
        assert all(p == 1 for p in predictions), "Should predict the mode"


class TestModelPersistence:
    """Tests for model saving and loading."""
    
    # Feature: ml-portfolio-project, Property 9: Pipeline Serialization Round-Trip
    def test_save_load_roundtrip(self):
        """Test that save/load produces identical model."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = train_baseline_model(X, y, 'regression')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            metadata = {'test_key': 'test_value'}
            
            save_model(model, filepath, metadata)
            loaded_model, loaded_metadata = load_model(filepath)
            
            # Check predictions are identical
            orig_preds = model.predict(X)
            loaded_preds = loaded_model.predict(X)
            
            np.testing.assert_array_equal(orig_preds, loaded_preds)
            assert loaded_metadata['test_key'] == 'test_value'
    
    # Feature: ml-portfolio-project, Property 27: Model Metadata Persistence
    def test_metadata_includes_required_fields(self):
        """Test that saved metadata includes required fields."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = train_baseline_model(X, y, 'regression')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            
            save_model(model, filepath, {'custom': 'data'})
            _, metadata = load_model(filepath)
            
            assert 'saved_at' in metadata, "Should include timestamp"
            assert 'model_type' in metadata, "Should include model type"
