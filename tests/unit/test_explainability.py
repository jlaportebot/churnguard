"""Tests for the explainability module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from churnguard.explainability import (
    ChurnExplainer,
    CustomerExplanation,
    GlobalExplanation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Simple binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "tenure": rng.exponential(36, n).clip(1, 120),
        "monthly_charges": rng.normal(65, 20, n).clip(18, 150),
        "total_charges": rng.normal(2000, 500, n).clip(0),
        "n_services": rng.binomial(5, 0.4, n),
    })
    y = pd.Series(rng.binomial(1, 0.3, n), name="churn")
    return X, y


@pytest.fixture
def fitted_lr(binary_data):
    """Fitted logistic regression model."""
    X, y = binary_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_rf(binary_data):
    """Fitted random forest model."""
    X, y = binary_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# CustomerExplanation tests
# ---------------------------------------------------------------------------

class TestCustomerExplanation:
    def test_top_features(self):
        expl = CustomerExplanation(
            customer_index=0,
            base_value=0.3,
            shap_values=np.array([0.1, -0.2, 0.05, 0.3]),
            feature_names=["a", "b", "c", "d"],
            predicted_probability=0.55,
        )
        top = expl.top_features(k=2)
        assert len(top) == 2
        # Top by |SHAP|: d (0.3), b (0.2)
        assert top[0][0] == "d"
        assert top[1][0] == "b"

    def test_risk_drivers(self):
        expl = CustomerExplanation(
            customer_index=0,
            base_value=0.3,
            shap_values=np.array([0.1, -0.2, 0.05, 0.3]),
            feature_names=["a", "b", "c", "d"],
            predicted_probability=0.55,
        )
        drivers = expl.risk_drivers(k=3)
        # Positive SHAP values: a=0.1, c=0.05, d=0.3
        assert len(drivers) <= 3
        assert all(v > 0 for _, v in drivers)
        # Highest positive should be d
        assert drivers[0][0] == "d"

    def test_protective_factors(self):
        expl = CustomerExplanation(
            customer_index=0,
            base_value=0.3,
            shap_values=np.array([0.1, -0.2, 0.05, -0.3]),
            feature_names=["a", "b", "c", "d"],
            predicted_probability=0.55,
        )
        factors = expl.protective_factors(k=2)
        assert len(factors) <= 2
        assert all(v < 0 for _, v in factors)

    def test_no_risk_drivers(self):
        expl = CustomerExplanation(
            customer_index=0,
            base_value=0.1,
            shap_values=np.array([-0.1, -0.2, -0.3]),
            feature_names=["a", "b", "c"],
            predicted_probability=0.1,
        )
        drivers = expl.risk_drivers()
        assert len(drivers) == 0

    def test_summary_output(self):
        expl = CustomerExplanation(
            customer_index=5,
            base_value=0.3,
            shap_values=np.array([0.2, -0.1, 0.05, 0.3]),
            feature_names=["tenure", "charges", "services", "support"],
            predicted_probability=0.75,
        )
        s = expl.summary()
        assert "Customer #5" in s
        assert "75.0%" in s


# ---------------------------------------------------------------------------
# GlobalExplanation tests
# ---------------------------------------------------------------------------

class TestGlobalExplanation:
    def test_top_features(self):
        ge = GlobalExplanation(
            feature_names=["a", "b", "c", "d"],
            mean_abs_shap=np.array([0.1, 0.5, 0.05, 0.3]),
            shap_values=np.random.randn(50, 4),
            base_value=0.3,
        )
        top = ge.top_features(k=3)
        assert len(top) == 3
        assert top[0][0] == "b"  # Highest mean_abs
        assert top[1][0] == "d"
        assert top[2][0] == "a"

    def test_to_dataframe(self):
        ge = GlobalExplanation(
            feature_names=["x", "y", "z"],
            mean_abs_shap=np.array([0.3, 0.1, 0.5]),
            shap_values=np.random.randn(10, 3),
            base_value=0.4,
        )
        df = ge.to_dataframe()
        assert len(df) == 3
        assert df.iloc[0]["feature"] == "z"  # Highest importance first

    def test_summary(self):
        ge = GlobalExplanation(
            feature_names=["a", "b", "c"],
            mean_abs_shap=np.array([0.3, 0.2, 0.1]),
            shap_values=np.random.randn(10, 3),
            base_value=0.35,
        )
        s = ge.summary(k=3)
        assert "Global Feature Importance" in s
        assert "a" in s


# ---------------------------------------------------------------------------
# ChurnExplainer tests (with shap)
# ---------------------------------------------------------------------------

class TestChurnExplainerKernel:
    """Test ChurnExplainer with KernelExplainer (logistic regression)."""

    def test_fit(self, binary_data, fitted_lr):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_lr, explainer_type="kernel")
        explainer.fit(X.iloc[:50], feature_names=list(X.columns))
        assert explainer._is_fitted

    def test_explain_global(self, binary_data, fitted_lr):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_lr, explainer_type="kernel")
        explainer.fit(X.iloc[:50], feature_names=list(X.columns))
        global_expl = explainer.explain_global(X.iloc[:30])
        assert isinstance(global_expl, GlobalExplanation)
        assert len(global_expl.feature_names) == 4
        assert global_expl.mean_abs_shap.shape == (4,)

    def test_explain_customer(self, binary_data, fitted_lr):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_lr, explainer_type="kernel")
        explainer.fit(X.iloc[:50], feature_names=list(X.columns))
        cust_expl = explainer.explain_customer(X.iloc[:10], customer_index=0)
        assert isinstance(cust_expl, CustomerExplanation)
        assert cust_expl.customer_index == 0
        assert 0 <= cust_expl.predicted_probability <= 1

    def test_not_fitted_raises(self, binary_data):
        X, _ = binary_data
        explainer = ChurnExplainer()
        with pytest.raises(RuntimeError, match="fitted"):
            explainer.explain_global(X)


class TestChurnExplainerTree:
    """Test ChurnExplainer with TreeExplainer (random forest)."""

    def test_auto_tree_selection(self, binary_data, fitted_rf):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_rf, explainer_type="auto")
        explainer.fit(X, feature_names=list(X.columns))
        # Should use TreeExplainer for RF
        assert "Tree" in type(explainer._explainer).__name__

    def test_explain_global_tree(self, binary_data, fitted_rf):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_rf, explainer_type="tree")
        explainer.fit(X, feature_names=list(X.columns))
        global_expl = explainer.explain_global(X.iloc[:50])
        assert isinstance(global_expl, GlobalExplanation)
        assert len(global_expl.feature_names) == 4

    def test_explain_customer_tree(self, binary_data, fitted_rf):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_rf, explainer_type="tree")
        explainer.fit(X, feature_names=list(X.columns))
        cust_expl = explainer.explain_customer(X.iloc[:5], customer_index=2)
        assert cust_expl.customer_index == 2

    def test_feature_importance_order(self, binary_data, fitted_rf):
        X, y = binary_data
        explainer = ChurnExplainer(model=fitted_rf, explainer_type="tree")
        explainer.fit(X, feature_names=list(X.columns))
        global_expl = explainer.explain_global(X.iloc[:50])
        top = global_expl.top_features(k=4)
        # All 4 features should be present
        assert len(top) == 4
        # Sorted descending by importance
        values = [v for _, v in top]
        assert values == sorted(values, reverse=True)
