"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.data import generate_sample_data
from churnguard.features import FeatureEngineer, FeatureSelector


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for feature engineering tests."""
        df = generate_sample_data(n_rows=200, random_state=42)
        X = df.drop(columns=["churn", "customer_id"])
        y = df["churn"]
        return X, y

    def test_fit_returns_self(self, sample_data):
        """Test that fit() returns self for chaining."""
        X, y = sample_data
        engineer = FeatureEngineer()
        result = engineer.fit(X)
        assert result is engineer

    def test_transform_output_shape(self, sample_data):
        """Test that transform produces a DataFrame with valid shape."""
        X, y = sample_data
        engineer = FeatureEngineer()
        X_tf = engineer.fit_transform(X)
        assert isinstance(X_tf, pd.DataFrame)
        assert len(X_tf) == len(X)

    def test_transform_before_fit_raises(self, sample_data):
        """Test that transform before fit raises RuntimeError."""
        X, _ = sample_data
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fitted"):
            engineer.transform(X)

    def test_numeric_imputation(self):
        """Test that NaN values in numeric columns are imputed."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "cat": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        })
        engineer = FeatureEngineer(numeric_impute_strategy="median")
        result = engineer.fit_transform(df)
        assert not result.isnull().any().any()

    def test_categorical_encoding(self):
        """Test one-hot encoding of categorical columns."""
        df = pd.DataFrame({
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        })
        engineer = FeatureEngineer()
        result = engineer.fit_transform(df)
        # "cat" has 2 unique values, one-hot with drop="if_binary" → 1 column
        assert result.shape[1] >= 2  # num + encoded cat

    def test_scaling(self, sample_data):
        """Test that scaling produces zero-mean, unit-variance for numeric columns."""
        X, y = sample_data
        engineer = FeatureEngineer(scaling="standard", generate_interactions=False)
        X_tf = engineer.fit_transform(X)
        # Only test columns produced by the ColumnTransformer (scaled numeric + one-hot)
        for col in X_tf.columns:
            # Skip one-hot encoded columns (they have non-zero mean by design)
            if any(cat in col for cat in ["contract_", "payment_", "internet_"]):
                continue
            mean = X_tf[col].mean()
            std = X_tf[col].std()
            if std > 0.1:
                assert abs(mean) < 0.5, f"Column {col} mean={mean}"
                assert 0.5 < std < 1.5, f"Column {col} std={std}"

    def test_no_scaling(self, sample_data):
        """Test that scaling='none' skips scaling."""
        X, y = sample_data
        engineer = FeatureEngineer(scaling="none")
        X_tf = engineer.fit_transform(X)
        assert isinstance(X_tf, pd.DataFrame)
        assert len(X_tf) == len(X)

    def test_frequency_encoding(self):
        """Test frequency encoding for high-cardinality columns."""
        df = pd.DataFrame({
            "num": range(100),
            "high_card": [f"val_{i % 25}" for i in range(100)],
        })
        engineer = FeatureEngineer(categorical_max_cardinality=10)
        result = engineer.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_interaction_features(self, sample_data):
        """Test interaction feature generation."""
        X, y = sample_data
        engineer = FeatureEngineer(generate_interactions=True)
        X_tf = engineer.fit_transform(X)
        # Should have some interaction features
        interaction_cols = [c for c in X_tf.columns if "_x_" in str(c)]
        assert len(interaction_cols) > 0

    def test_no_interactions(self, sample_data):
        """Test disabling interaction features."""
        X, y = sample_data
        engineer = FeatureEngineer(generate_interactions=False)
        X_tf = engineer.fit_transform(X)
        interaction_cols = [c for c in X_tf.columns if "_x_" in str(c)]
        assert len(interaction_cols) == 0

    def test_feature_names_property(self, sample_data):
        """Test feature_names property after fitting."""
        X, y = sample_data
        engineer = FeatureEngineer()
        engineer.fit(X)
        names = engineer.feature_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_feature_names_before_fit_raises(self):
        """Test that accessing feature_names before fit raises."""
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = engineer.feature_names

    def test_n_features_out(self, sample_data):
        """Test n_features_out property."""
        X, y = sample_data
        engineer = FeatureEngineer()
        engineer.fit(X)
        assert engineer.n_features_out == len(engineer.feature_names)

    def test_transform_new_data(self, sample_data):
        """Test that transform works on new data with the same schema."""
        X, y = sample_data
        engineer = FeatureEngineer()
        engineer.fit(X)
        # Transform a subset of the data
        X_new = X.iloc[:10]
        X_new_tf = engineer.transform(X_new)
        assert len(X_new_tf) == 10


class TestFeatureSelector:
    """Tests for FeatureSelector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample numeric data."""
        np.random.seed(42)
        return pd.DataFrame({
            "useful": np.random.randn(100),
            "also_useful": np.random.randn(100),
            "constant": np.zeros(100),  # Zero variance
            "near_constant": np.ones(100) * 0.001,  # Near-zero variance
        })

    def test_variance_selection(self, sample_data):
        """Test variance-based feature selection."""
        selector = FeatureSelector(method="variance", threshold=0.01)
        result = selector.fit_transform(sample_data)
        assert "constant" not in result.columns
        assert "near_constant" not in result.columns
        assert "useful" in result.columns

    def test_max_features(self, sample_data):
        """Test max_features limit."""
        selector = FeatureSelector(method="variance", threshold=0.0, max_features=2)
        result = selector.fit_transform(sample_data)
        assert result.shape[1] <= 2

    def test_selected_features_property(self, sample_data):
        """Test selected_features property."""
        selector = FeatureSelector(method="variance", threshold=0.01)
        selector.fit(sample_data)
        features = selector.selected_features
        assert isinstance(features, list)
        assert "useful" in features

    def test_correlation_selection(self):
        """Test correlation-based feature selection."""
        np.random.seed(42)
        X = pd.DataFrame({
            "good": np.random.randn(100),
            "bad": np.random.randn(100) * 0.01,
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        # Make 'good' correlated with y
        X["good"] = X["good"] + y * 2

        selector = FeatureSelector(method="correlation", threshold=0.1)
        result = selector.fit_transform(X, y)
        assert "good" in result.columns
