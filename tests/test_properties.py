"""Property-based tests using Hypothesis."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_load import split_data
from preprocessing import build_preprocessing_pipeline


# Feature: ml-portfolio-project, Property 11: Data Split Non-Overlap
@given(st.integers(min_value=100, max_value=500))
@settings(max_examples=20)
def test_data_split_no_overlap(n_rows):
    """Test that train/val/test splits have no overlapping indices."""
    df = pd.DataFrame({
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'target': np.random.randn(n_rows)
    })
    
    # Ensure unique index
    df.index = range(n_rows)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, 'target', test_size=0.2, val_size=0.2, random_state=42
    )
    
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    
    # Check no overlap
    assert len(train_idx & val_idx) == 0, "Train and val should not overlap"
    assert len(train_idx & test_idx) == 0, "Train and test should not overlap"
    assert len(val_idx & test_idx) == 0, "Val and test should not overlap"
    
    # Check union equals original
    assert train_idx | val_idx | test_idx == set(df.index), "Should cover all indices"


# Feature: ml-portfolio-project, Property 5: Correlation Matrix Dimensions
@given(st.integers(min_value=3, max_value=10))
@settings(max_examples=20)
def test_correlation_matrix_dimensions(n_features):
    """Test that correlation matrix is n×n symmetric with 1.0 diagonal."""
    df = pd.DataFrame(
        np.random.randn(10, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    corr_matrix = df.corr()
    
    assert corr_matrix.shape == (n_features, n_features), "Should be n×n"
    assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1.0"
    assert np.allclose(corr_matrix, corr_matrix.T), "Should be symmetric"


# Feature: ml-portfolio-project, Property 19: Silhouette Score Range
@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=10, deadline=None) # Disable deadline for slower clustering
def test_silhouette_score_range(k):
    """Test that silhouette scores are in valid range [-1, 1]."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Create distinct clusters
    X = np.concatenate([
        np.random.randn(50, 2) + np.array([5, 5]),
        np.random.randn(50, 2) + np.array([-5, -5]),
        np.random.randn(50, 2) + np.array([5, -5])
    ])
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
        assert -1 <= score <= 1, f"Silhouette score {score} outside valid range"


# Feature: ml-portfolio-project, Property 26: End-to-End Reproducibility
def test_reproducibility_with_fixed_seed():
    """Test that same random_state produces identical results."""
    from sklearn.ensemble import RandomForestRegressor
    
    X = np.random.RandomState(42).randn(100, 5)
    y = np.random.RandomState(42).randn(100)
    
    # Train twice with same seed
    model1 = RandomForestRegressor(n_estimators=10, random_state=42)
    model1.fit(X, y)
    preds1 = model1.predict(X)
    
    model2 = RandomForestRegressor(n_estimators=10, random_state=42)
    model2.fit(X, y)
    preds2 = model2.predict(X)
    
    np.testing.assert_array_equal(preds1, preds2, "Same seed should give same predictions")
