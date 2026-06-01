"""Tests for the model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.data import generate_sample_data
from churnguard.features import FeatureEngineer
from churnguard.models import (
    LogisticChurnModel,
    RandomForestChurnModel,
    GradientBoostingChurnModel,
    ModelRegistry,
)
from churnguard.models.base import ModelResult


@pytest.fixture(scope="module")
def prepared_data():
    """Create and prepare a dataset for model testing."""
    df = generate_sample_data(n_rows=300, churn_rate=0.25, random_state=42)
    from churnguard.data import DataLoader
    loader = DataLoader.__new__(DataLoader)
    loader._df = df
    loader._target_column_resolved = "churn"
    loader._id_columns_resolved = ["customer_id"]

    X, y = loader.get_features_and_target()

    engineer = FeatureEngineer(generate_interactions=False)
    X_tf = engineer.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tf, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X_tf.columns)


class TestLogisticChurnModel:
    """Tests for LogisticChurnModel."""

    def test_name(self):
        model = LogisticChurnModel()
        assert model.name == "Logistic Regression"

    def test_default_params(self):
        model = LogisticChurnModel()
        params = model.default_params
        assert "C" in params
        assert "max_iter" in params

    def test_fit_and_predict(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = LogisticChurnModel()
        model.fit(X_train, y_train)
        assert model.is_fitted

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = LogisticChurnModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert len(proba) == len(y_test)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_evaluate(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = LogisticChurnModel()
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert isinstance(result, ModelResult)
        assert result.model_name == "Logistic Regression"
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1 <= 1
        assert 0 <= result.roc_auc <= 1

    def test_predict_before_fit_raises(self):
        model = LogisticChurnModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(pd.DataFrame({"a": [1]}))

    def test_custom_params(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = LogisticChurnModel(C=0.1, max_iter=500)
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert result.accuracy > 0.5  # Should be better than random

    def test_feature_importance(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = LogisticChurnModel()
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert len(result.feature_importance) > 0


class TestRandomForestChurnModel:
    """Tests for RandomForestChurnModel."""

    def test_name(self):
        model = RandomForestChurnModel()
        assert model.name == "Random Forest"

    def test_fit_and_predict(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = RandomForestChurnModel(n_estimators=10, max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_evaluate(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = RandomForestChurnModel(n_estimators=10, max_depth=5)
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert result.roc_auc > 0.5  # Better than random

    def test_feature_importance(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = RandomForestChurnModel(n_estimators=10, max_depth=5)
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert len(result.feature_importance) > 0
        # Top feature should have positive importance
        top = max(result.feature_importance.values())
        assert top > 0


class TestGradientBoostingChurnModel:
    """Tests for GradientBoostingChurnModel."""

    def test_name(self):
        model = GradientBoostingChurnModel()
        assert model.name == "Gradient Boosting"

    def test_fit_and_predict(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = GradientBoostingChurnModel(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_evaluate(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        model = GradientBoostingChurnModel(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test, feature_names=feat_names)
        assert result.roc_auc > 0.5


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_available_models(self):
        registry = ModelRegistry()
        assert "logistic" in registry.available_models
        assert "random_forest" in registry.available_models
        assert "gradient_boosting" in registry.available_models

    def test_custom_model_selection(self):
        registry = ModelRegistry(models=["logistic"])
        assert registry.available_models == ["logistic"]

    def test_get_model(self):
        registry = ModelRegistry()
        model = registry.get_model("logistic")
        assert isinstance(model, LogisticChurnModel)

    def test_get_unknown_model_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get_model("nonexistent")

    def test_train_and_evaluate(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        registry = ModelRegistry(models=["logistic"])
        result = registry.train_and_evaluate(
            "logistic", X_train, X_test, y_train, y_test, feature_names=feat_names
        )
        assert result.f1 > 0

    def test_compare_all(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        # Use small models for speed
        registry = ModelRegistry(models=["logistic", "random_forest", "gradient_boosting"])
        # Override model params for speed
        registry._model_instances["random_forest"] = RandomForestChurnModel(n_estimators=10, max_depth=5)
        registry._model_instances["gradient_boosting"] = GradientBoostingChurnModel(n_estimators=10, max_depth=3)

        results = registry.compare_all(
            X_train, X_test, y_train, y_test, feature_names=feat_names
        )
        assert len(results) == 3
        assert all(r.f1 > 0 for r in results.values())

    def test_get_best(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        registry = ModelRegistry(models=["logistic", "random_forest"])
        registry._model_instances["random_forest"] = RandomForestChurnModel(n_estimators=10, max_depth=5)
        results = registry.compare_all(
            X_train, X_test, y_train, y_test, feature_names=feat_names
        )
        best = registry.get_best(results, metric="f1")
        assert best.f1 > 0

    def test_comparison_table(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        registry = ModelRegistry(models=["logistic", "random_forest"])
        registry._model_instances["random_forest"] = RandomForestChurnModel(n_estimators=10, max_depth=5)
        results = registry.compare_all(
            X_train, X_test, y_train, y_test, feature_names=feat_names
        )
        table = registry.comparison_table(results)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2

    def test_evaluate_untrained_model_raises(self, prepared_data):
        X_train, X_test, y_train, y_test, feat_names = prepared_data
        registry = ModelRegistry(models=["logistic"])
        with pytest.raises(RuntimeError, match="trained"):
            registry.evaluate_model("logistic", X_test, y_test)


class TestModelResult:
    """Tests for ModelResult dataclass."""

    def test_summary(self):
        result = ModelResult(
            model_name="TestModel",
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.90,
            pr_auc=0.82,
            training_time_seconds=1.5,
            cv_scores=[0.75, 0.78, 0.76],
        )
        summary = result.summary()
        assert "TestModel" in summary
        assert "0.85" in summary

    def test_to_dict(self):
        result = ModelResult(
            model_name="TestModel",
            accuracy=0.85,
            f1=0.77,
            feature_importance={"feat_a": 0.5, "feat_b": 0.3},
            cv_scores=[0.75, 0.78],
        )
        d = result.to_dict()
        assert d["model_name"] == "TestModel"
        assert d["accuracy"] == 0.85
        assert "top_5_features" in d
