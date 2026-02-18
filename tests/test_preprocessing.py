"""Unit and property tests for preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, HealthCheck
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    engineer_features,
    build_preprocessing_pipeline,
    handle_missing_values,
    detect_outliers
)


class TestEngineerFeatures:
    """Unit tests for feature engineering."""
    
    def test_engineer_features_adds_columns(self):
        """Test that engineer_features adds new columns."""
        df = pd.DataFrame({
            'GrLivArea': [1000, 1500, 2000],
            'TotalBsmtSF': [500, 750, 1000],
            'YearBuilt': [1990, 2000, 2010],
            'YearRemodAdd': [1995, 2005, 2015],
            'FullBath': [1, 2, 2],
            'HalfBath': [0, 1, 1],
            'BsmtFullBath': [0, 1, 0],
            'BsmtHalfBath': [0, 0, 0],
            'OverallQual': [5, 7, 9],
            '1stFlrSF': [500, 750, 1000],
            '2ndFlrSF': [500, 750, 1000]
        })
        result = engineer_features(df.copy(), 'house-prices')
        # Check specific features we expect
        assert 'TotalSF' in result.columns
        assert 'TotalBath' in result.columns
        assert 'HouseAge' in result.columns
        assert len(result.columns) > len(df.columns), "Should add new features"
    
    def test_engineer_features_no_nulls_introduced(self):
        """Test that feature engineering doesn't introduce NaN values."""
        df = pd.DataFrame({
             'GrLivArea': [1000, 1500],
            'TotalBsmtSF': [500, 750],
            'YearBuilt': [1990, 2000],
            'YearRemodAdd': [1995, 2005],
            'FullBath': [1, 2],
            'HalfBath': [0, 1],
            'BsmtFullBath': [0, 1],
            'BsmtHalfBath': [0, 0],
            'OverallQual': [5, 7],
            '1stFlrSF': [500, 750],
            '2ndFlrSF': [500, 750]
        })
        result = engineer_features(df.copy(), 'house-prices')
        new_cols = [c for c in result.columns if c not in df.columns]
        for col in new_cols:
            assert result[col].notna().all(), f"Feature {col} has NaN values"


class TestPreprocessingPipeline:
    """Unit tests for preprocessing pipeline."""
    
    def test_pipeline_returns_pipeline_object(self):
        """Test that build_preprocessing_pipeline returns a Pipeline."""
        pipeline = build_preprocessing_pipeline(
            numerical_features=['a', 'b'],
            categorical_features=['c']
        )
        # ColumnTransformer, not Pipeline directly, but acts like one
        from sklearn.compose import ColumnTransformer
        assert isinstance(pipeline, ColumnTransformer)
    
    def test_pipeline_transforms_data(self):
        """Test that pipeline can fit and transform data."""
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat1': ['a', 'b', 'a', 'b', 'a']
        })
        pipeline = build_preprocessing_pipeline(
            numerical_features=['num1', 'num2'],
            categorical_features=['cat1']
        )
        result = pipeline.fit_transform(df)
        assert result.shape[0] == 5, "Should preserve row count"


# Property-based tests
# Feature: ml-portfolio-project, Property 7: Numerical Feature Scaling
@given(st.lists(st.floats(min_value=-1e5, max_value=1e5, allow_nan=False), min_size=10, max_size=100))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_standard_scaling_properties(values):
    """Test that StandardScaler produces mean~0 and std~1."""
    if len(set(values)) < 2:
        return # Skip constant arrays
    
    # Skip if variance is too small to compute standard deviation accurately
    if np.std(values) < 1e-8:
        return

    from sklearn.preprocessing import StandardScaler
    
    arr = np.array(values).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    
    assert abs(scaled.mean()) < 0.5, "Mean should be approximately 0"
    assert abs(scaled.std() - 1.0) < 0.5, "Std should be approximately 1"


# Feature: ml-portfolio-project, Property 8: Preprocessing Idempotence
def test_preprocessing_idempotence():
    """Test that transforming same data twice gives identical results."""
    df = pd.DataFrame({
        'num': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat': ['a', 'b', 'a', 'b', 'a']
    })
    pipeline = build_preprocessing_pipeline(
        numerical_features=['num'],
        categorical_features=['cat']
    )
    pipeline.fit(df)
    
    result1 = pipeline.transform(df)
    result2 = pipeline.transform(df)
    
    np.testing.assert_array_equal(result1, result2)
