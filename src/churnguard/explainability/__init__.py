"""Model explainability module using SHAP values.

Provides global and per-customer explanations for churn prediction models:
- Global feature importance with SHAP values
- Per-customer SHAP explanations (why was this customer flagged?)
- Text summaries of key drivers

Requires the ``shap`` package (install with: ``pip install shap``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import shap — it's an optional dependency
_shap = None


def _get_shap():
    """Lazily import shap, raising ImportError with a helpful message."""
    global _shap
    if _shap is None:
        try:
            import shap as _shap_module

            _shap = _shap_module
        except ImportError:
            raise ImportError(
                "The 'shap' package is required for explainability. "
                "Install it with: pip install shap"
            ) from None
    return _shap


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CustomerExplanation:
    """SHAP explanation for a single customer.

    Attributes:
        customer_index: Row index of the customer in the dataset.
        base_value: Expected model output (average prediction).
        shap_values: SHAP value for each feature.
        feature_names: Names of features in the same order as shap_values.
        predicted_probability: Model's predicted churn probability.
    """

    customer_index: int
    base_value: float
    shap_values: np.ndarray
    feature_names: list[str]
    predicted_probability: float

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k features by absolute SHAP value.

        Parameters
        ----------
        k : int
            Number of top features to return.

        Returns
        -------
        list of (feature_name, shap_value) tuples, sorted by |SHAP| descending.
        """
        abs_vals = np.abs(self.shap_values)
        top_indices = np.argsort(abs_vals)[::-1][:k]
        return [(self.feature_names[i], float(self.shap_values[i])) for i in top_indices]

    def risk_drivers(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k features that *increase* churn risk (positive SHAP).

        Parameters
        ----------
        k : int
            Number of risk drivers to return.

        Returns
        -------
        list of (feature_name, shap_value) tuples, sorted by SHAP descending.
        """
        positive_mask = self.shap_values > 0
        positive_vals = self.shap_values[positive_mask]
        positive_names = [
            self.feature_names[i] for i in range(len(self.feature_names)) if positive_mask[i]
        ]

        if len(positive_vals) == 0:
            return []

        sorted_idx = np.argsort(positive_vals)[::-1][:k]
        return [(positive_names[i], float(positive_vals[i])) for i in sorted_idx]

    def protective_factors(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k features that *decrease* churn risk (negative SHAP).

        Parameters
        ----------
        k : int
            Number of protective factors to return.

        Returns
        -------
        list of (feature_name, shap_value) tuples, sorted by SHAP ascending.
        """
        negative_mask = self.shap_values < 0
        negative_vals = self.shap_values[negative_mask]
        negative_names = [
            self.feature_names[i] for i in range(len(self.feature_names)) if negative_mask[i]
        ]

        if len(negative_vals) == 0:
            return []

        sorted_idx = np.argsort(negative_vals)[:k]
        return [(negative_names[i], float(negative_vals[i])) for i in sorted_idx]

    def summary(self) -> str:
        """Human-readable explanation summary."""
        lines = [
            f"Customer #{self.customer_index} — Predicted churn risk: {self.predicted_probability:.1%}",
            f"Base value (avg prediction): {self.base_value:.4f}",
            "",
            "Top risk drivers:",
        ]
        for name, val in self.risk_drivers(k=3):
            lines.append(f"  + {name}: {val:+.4f}")

        lines.append("")
        lines.append("Top protective factors:")
        for name, val in self.protective_factors(k=3):
            lines.append(f"  - {name}: {val:+.4f}")

        return "\n".join(lines)


@dataclass
class GlobalExplanation:
    """Global SHAP explanation across all customers.

    Attributes:
        feature_names: Names of features.
        mean_abs_shap: Mean |SHAP| value per feature (global importance).
        shap_values: Full SHAP matrix (n_samples × n_features).
        base_value: Expected model output.
    """

    feature_names: list[str]
    mean_abs_shap: np.ndarray
    shap_values: np.ndarray
    base_value: float

    def top_features(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k features by mean absolute SHAP value.

        Parameters
        ----------
        k : int
            Number of top features.

        Returns
        -------
        list of (feature_name, mean_abs_shap) tuples.
        """
        sorted_idx = np.argsort(self.mean_abs_shap)[::-1][:k]
        return [(self.feature_names[i], float(self.mean_abs_shap[i])) for i in sorted_idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert global explanation to a DataFrame."""
        sorted_idx = np.argsort(self.mean_abs_shap)[::-1]
        return pd.DataFrame(
            {
                "feature": [self.feature_names[i] for i in sorted_idx],
                "importance": [float(self.mean_abs_shap[i]) for i in sorted_idx],
            }
        )

    def summary(self, k: int = 10) -> str:
        """Human-readable summary of global feature importance."""
        lines = ["=== Global Feature Importance (SHAP) ==="]
        lines.append(f"Base value: {self.base_value:.4f}")
        lines.append("")
        for rank, (name, val) in enumerate(self.top_features(k), 1):
            lines.append(f"  {rank:>2d}. {name:<30s} {val:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ChurnExplainer
# ---------------------------------------------------------------------------


class ChurnExplainer:
    """SHAP-based explainer for churn prediction models.

    Wraps any fitted sklearn-compatible model and computes SHAP values
    for global and local explanations.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model with ``predict_proba`` method.
    background_data : pd.DataFrame or np.ndarray, optional
        Background dataset for KernelExplainer. If None, uses a subsample
        of the training data (call ``fit`` first).
    n_background_samples : int
        Number of background samples for KernelExplainer (default 100).
    explainer_type : str
        'auto', 'tree', or 'kernel'. 'auto' picks TreeExplainer for
        tree-based models, KernelExplainer otherwise.
    """

    def __init__(
        self,
        model: Any = None,
        background_data: pd.DataFrame | None = None,
        n_background_samples: int = 100,
        explainer_type: str = "auto",
    ) -> None:
        self.model = model
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        self.explainer_type = explainer_type
        self._explainer = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        model: Any = None,
        feature_names: list[str] | None = None,
    ) -> ChurnExplainer:
        """Fit the explainer on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (used as background data if not pre-set).
        model : sklearn estimator, optional
            Override the model passed at init.
        feature_names : list of str, optional
            Feature names. Defaults to DataFrame columns.

        Returns
        -------
        self
        """
        shap = _get_shap()

        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Pass model at init or to fit().")

        if feature_names is not None:
            self._feature_names = feature_names
        elif hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Prepare background data
        if self.background_data is not None:
            bg = self.background_data
        else:
            n_bg = min(self.n_background_samples, len(X))
            bg = X.iloc[:n_bg] if hasattr(X, "iloc") else X[:n_bg]

        # Choose explainer
        if self.explainer_type == "auto":
            model_type = type(self.model).__name__.lower()
            tree_models = {
                "randomforestclassifier",
                "gradientboostingclassifier",
                "decisiontreeclassifier",
                "extratreesclassifier",
            }
            if any(t in model_type for t in tree_models):
                self._explainer = shap.TreeExplainer(self.model)
            else:
                self._explainer = shap.KernelExplainer(self.model.predict_proba, bg)
        elif self.explainer_type == "tree":
            self._explainer = shap.TreeExplainer(self.model)
        else:
            self._explainer = shap.KernelExplainer(self.model.predict_proba, bg)

        self._is_fitted = True
        logger.info("ChurnExplainer fitted with %s", type(self._explainer).__name__)
        return self

    def explain_global(
        self,
        X: pd.DataFrame,
    ) -> GlobalExplanation:
        """Compute global feature importance via SHAP values.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to explain (typically the test set).

        Returns
        -------
        GlobalExplanation
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer must be fitted first. Call fit().")

        shap_values = self._compute_shap_values(X)
        sv = self._extract_positive_class_shap(shap_values)

        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        mean_abs = np.mean(np.abs(sv), axis=0)
        # Flatten in case it's still multi-dimensional
        mean_abs = mean_abs.flatten()
        base_value = float(self._get_base_value())

        n_features = sv.shape[1] if sv.ndim >= 2 else len(sv)
        return GlobalExplanation(
            feature_names=self._feature_names[:n_features],
            mean_abs_shap=mean_abs,
            shap_values=sv,
            base_value=base_value,
        )

    def explain_customer(
        self,
        X: pd.DataFrame,
        customer_index: int = 0,
    ) -> CustomerExplanation:
        """Compute SHAP explanation for a single customer.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset containing the customer.
        customer_index : int
            Row index of the customer in X.

        Returns
        -------
        CustomerExplanation
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer must be fitted first. Call fit().")

        # Get SHAP values for just this customer
        single = X.iloc[[customer_index]] if hasattr(X, "iloc") else X[[customer_index]]

        shap_values = self._compute_shap_values(single)
        sv = self._extract_positive_class_shap(shap_values)

        # Get the single-customer row
        sv_row = sv[0].flatten() if sv.ndim >= 2 else np.array(sv).flatten()

        base_value = float(self._get_base_value())

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(single)[:, 1][0])
        else:
            proba = float(self.model.predict(single)[0])

        return CustomerExplanation(
            customer_index=customer_index,
            base_value=base_value,
            shap_values=sv_row,
            feature_names=self._feature_names[: len(sv_row)],
            predicted_probability=proba,
        )

    def _compute_shap_values(self, X: Any) -> Any:
        """Compute SHAP values using the fitted explainer."""
        # Subsample for KernelExplainer (can be slow on large datasets)
        if hasattr(self._explainer, "expected_value") and not hasattr(
            self._explainer, "tree_limit"
        ):
            # KernelExplainer — limit to 200 samples for speed
            X_sample = X.iloc[:200] if hasattr(X, "iloc") and len(X) > 200 else X
        else:
            X_sample = X

        try:
            return self._explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning("SHAP computation failed: %s. Trying with smaller sample.", e)
            # Fallback: use even smaller sample
            small = X.iloc[:50] if hasattr(X, "iloc") else X[:50]
            return self._explainer.shap_values(small)

    def _extract_positive_class_shap(self, shap_values: Any) -> np.ndarray:
        """Extract SHAP values for the positive class from various return formats.

        Handles list returns (older shap), 3D arrays (newer shap),
        and Explanation objects.
        """
        if isinstance(shap_values, list):
            # Older shap versions return list for multi-output
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return np.array(sv)

        sv = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

        # 3D array: (n_samples, n_features, n_classes) → take positive class
        if sv.ndim == 3 and sv.shape[2] >= 2:
            sv = sv[:, :, 1]

        return sv

    def _get_base_value(self) -> float:
        """Extract the base value (expected value) from the explainer."""
        try:
            ev = self._explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                return float(ev[-1])  # positive class for binary
            return float(ev)
        except (AttributeError, TypeError):
            return 0.0


__all__ = [
    "ChurnExplainer",
    "CustomerExplanation",
    "GlobalExplanation",
]
