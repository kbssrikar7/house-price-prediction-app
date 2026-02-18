"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_load import (
    validate_dataset,
    split_data,
    get_dataset_config
)


class TestValidateDataset:
    """Tests for dataset validation."""

    def test_valid_dataset(self):
        """Test validation passes for valid dataset."""
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100),
            'C': ['a'] * 100
        })
        result = validate_dataset(df)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0

    def test_insufficient_rows(self):
        """Test validation fails for small dataset."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = validate_dataset(df)
        assert result['is_valid'] is False
        assert any('rows' in e.lower() for e in result['errors'])

    def test_insufficient_columns(self):
        """Test validation fails for single column."""
        df = pd.DataFrame({'A': range(100)})
        result = validate_dataset(df)
        assert result['is_valid'] is False

    def test_returns_expected_keys(self):
        """Test that validate_dataset returns all expected keys."""
        df = pd.DataFrame({'A': range(100), 'B': range(100)})
        result = validate_dataset(df)
        expected_keys = {'is_valid', 'errors', 'warnings', 'n_rows', 'n_cols', 'dtypes'}
        assert set(result.keys()) == expected_keys


class TestSplitData:
    """Tests for data splitting."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })

    def test_split_sizes(self, sample_df):
        """Test split produces correct sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            sample_df, 'target', test_size=0.2, val_size=0.1
        )

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 1000
        assert len(X_test) == pytest.approx(200, abs=10)
        assert len(X_val) == pytest.approx(100, abs=15)

    def test_no_overlap(self, sample_df):
        """Test train/val/test sets don't overlap."""
        X_train, X_val, X_test, _, _, _ = split_data(
            sample_df, 'target'
        )

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_target_not_in_features(self, sample_df):
        """Test that target column is not in feature DataFrames."""
        X_train, X_val, X_test, _, _, _ = split_data(
            sample_df, 'target'
        )

        assert 'target' not in X_train.columns
        assert 'target' not in X_val.columns
        assert 'target' not in X_test.columns

    def test_reproducibility(self, sample_df):
        """Test that splits are reproducible with same random_state."""
        result1 = split_data(sample_df, 'target', random_state=42)
        result2 = split_data(sample_df, 'target', random_state=42)

        pd.testing.assert_frame_equal(result1[0], result2[0])
        pd.testing.assert_series_equal(result1[3], result2[3])


class TestGetDatasetConfig:
    """Tests for dataset config retrieval."""

    def test_valid_dataset_name(self):
        """Test config retrieval for valid dataset."""
        config = get_dataset_config('house-prices-advanced-regression-techniques')
        assert 'target_column' in config
        assert 'task_type' in config
        assert config['task_type'] == 'regression'

    def test_invalid_dataset_name(self):
        """Test error for unknown dataset."""
        with pytest.raises(ValueError, match='Unknown dataset'):
            get_dataset_config('nonexistent-dataset')
