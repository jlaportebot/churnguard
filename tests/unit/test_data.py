"""Tests for the data loading module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnguard.data import DataLoader, DataValidationError, generate_sample_data


class TestDataLoader:
    """Tests for DataLoader class."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample CSV file."""
        df = generate_sample_data(n_rows=200, churn_rate=0.25, random_state=42)
        path = tmp_path / "sample.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def minimal_csv(self, tmp_path: Path) -> Path:
        """Create a minimal valid CSV."""
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_b": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "churn": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        })
        path = tmp_path / "minimal.csv"
        df.to_csv(path, index=False)
        return path

    def test_load_csv(self, sample_csv: Path):
        """Test loading a CSV file."""
        loader = DataLoader(sample_csv)
        df = loader.df
        assert len(df) == 200
        assert "churn" in df.columns

    def test_validate_auto_detect_target(self, sample_csv: Path):
        """Test auto-detection of target column."""
        loader = DataLoader(sample_csv)
        loader.validate()
        assert loader.target_name == "churn"

    def test_validate_explicit_target(self, sample_csv: Path):
        """Test explicit target column specification."""
        loader = DataLoader(sample_csv, target_column="churn")
        loader.validate()
        assert loader.target_name == "churn"

    def test_validate_invalid_target(self, sample_csv: Path):
        """Test that an invalid target column raises an error."""
        loader = DataLoader(sample_csv, target_column="nonexistent")
        with pytest.raises(DataValidationError, match="not found"):
            loader.validate()

    def test_split(self, sample_csv: Path):
        """Test train/test split."""
        loader = DataLoader(sample_csv, target_column="churn", test_size=0.2)
        X_train, X_test, y_train, y_test = loader.split()
        assert len(X_train) == 160
        assert len(X_test) == 40
        assert len(y_train) == 160
        assert len(y_test) == 40

    def test_get_features_and_target(self, sample_csv: Path):
        """Test feature/target separation."""
        loader = DataLoader(sample_csv, target_column="churn")
        X, y = loader.get_features_and_target()
        assert "churn" not in X.columns
        assert "customer_id" not in X.columns  # ID column should be dropped
        assert y.name == "churn"

    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        loader = DataLoader("/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            _ = loader.df

    def test_empty_dataset(self, tmp_path: Path):
        """Test that empty dataset raises DataValidationError."""
        path = tmp_path / "empty.csv"
        pd.DataFrame({"a": [], "churn": []}).to_csv(path, index=False)
        loader = DataLoader(path, target_column="churn")
        with pytest.raises(DataValidationError, match="empty"):
            loader.validate()

    def test_single_class_target(self, tmp_path: Path):
        """Test that single-class target raises DataValidationError."""
        path = tmp_path / "single_class.csv"
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "churn": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        df.to_csv(path, index=False)
        loader = DataLoader(path, target_column="churn")
        with pytest.raises(DataValidationError, match="unique value"):
            loader.validate()

    def test_string_target_encoding(self, tmp_path: Path):
        """Test that string targets are auto-encoded."""
        path = tmp_path / "string_target.csv"
        df = pd.DataFrame({
            "feature": np.random.randn(20),
            "churn": ["no", "yes"] * 10,
        })
        df.to_csv(path, index=False)
        loader = DataLoader(path, target_column="churn")
        X, y = loader.get_features_and_target()
        assert set(y.unique()) == {0, 1}

    def test_custom_test_size(self, sample_csv: Path):
        """Test custom test size."""
        loader = DataLoader(sample_csv, target_column="churn", test_size=0.3)
        X_train, X_test, y_train, y_test = loader.split(test_size=0.3)
        assert len(X_test) == 60

    def test_auto_detect_id_columns(self, sample_csv: Path):
        """Test auto-detection of ID columns."""
        loader = DataLoader(sample_csv)
        loader.validate()
        assert "customer_id" in loader.detected_id_columns

    def test_minimal_dataset(self, minimal_csv: Path):
        """Test with a minimal valid dataset."""
        loader = DataLoader(minimal_csv, target_column="churn")
        X_train, X_test, y_train, y_test = loader.split(test_size=0.2)
        assert len(X_train) + len(X_test) == 10
        assert set(y_train.unique()).issubset({0, 1})

    def test_drop_na_target(self, tmp_path: Path):
        """Test dropping rows with missing target."""
        path = tmp_path / "with_na.csv"
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "churn": [0, 1, np.nan, 0, 1, np.nan, 0, 1, 0, 1],
        })
        df.to_csv(path, index=False)
        loader = DataLoader(path, target_column="churn", drop_na_target=True)
        loader.validate()
        assert len(loader.df) == 8  # 2 rows dropped


class TestGenerateSampleData:
    """Tests for sample data generation."""

    def test_basic_generation(self):
        """Test basic sample data generation."""
        df = generate_sample_data(n_rows=100)
        assert len(df) == 100
        assert "churn" in df.columns

    def test_churn_rate(self):
        """Test approximate churn rate."""
        df = generate_sample_data(n_rows=5000, churn_rate=0.3, random_state=42)
        actual_rate = df["churn"].mean()
        assert 0.15 < actual_rate < 0.5  # Allow generous variance

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        df1 = generate_sample_data(n_rows=50, random_state=42)
        df2 = generate_sample_data(n_rows=50, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        df1 = generate_sample_data(n_rows=50, random_state=42)
        df2 = generate_sample_data(n_rows=50, random_state=99)
        assert not df1.equals(df2)

    def test_has_missing_values(self):
        """Test that sample data includes some NaN values."""
        df = generate_sample_data(n_rows=1000, random_state=42)
        assert df.isnull().any().any()

    def test_column_types(self):
        """Test expected column types."""
        df = generate_sample_data(n_rows=100)
        assert df["tenure"].dtype in [np.int64, np.float64]
        assert pd.api.types.is_string_dtype(df["contract"]) or df["contract"].dtype == object
        assert df["churn"].dtype in [np.int64, np.int32, np.float64]

    def test_customer_id_format(self):
        """Test customer ID format."""
        df = generate_sample_data(n_rows=10)
        assert df["customer_id"].iloc[0].startswith("CUST-")
