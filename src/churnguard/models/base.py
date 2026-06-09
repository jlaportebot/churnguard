"""Base model interface and result container."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for model evaluation results.

    Attributes
    ----------
    model_name : str
        Name of the model.
    accuracy : float
        Accuracy score.
    precision : float
        Precision score (positive class).
    recall : float
        Recall score (positive class).
    f1 : float
        F1 score (positive class).
    roc_auc : float
        Area under the ROC curve.
    pr_auc : float
        Area under the Precision-Recall curve.
    confusion_matrix : np.ndarray
        2x2 confusion matrix.
    feature_importance : dict[str, float]
        Feature name → importance score.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    training_time_seconds : float
        Wall-clock training time.
    best_params : dict[str, Any]
        Best hyperparameters found (if tuned).
    cv_scores : list[float]
        Cross-validation scores.
    """

    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    training_time_seconds: float = 0.0
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Model: {self.model_name}",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1:        {self.f1:.4f}",
            f"  ROC AUC:   {self.roc_auc:.4f}",
            f"  PR AUC:    {self.pr_auc:.4f}",
            f"  Train time: {self.training_time_seconds:.2f}s",
        ]
        if self.cv_scores:
            lines.append(
                f"  CV mean:   {np.mean(self.cv_scores):.4f} (±{np.std(self.cv_scores):.4f})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary (excluding arrays)."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "training_time_seconds": self.training_time_seconds,
            "best_params": self.best_params,
            "cv_scores_mean": float(np.mean(self.cv_scores)) if self.cv_scores else None,
            "cv_scores_std": float(np.std(self.cv_scores)) if self.cv_scores else None,
            "n_features": len(self.feature_importance),
            "top_5_features": dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }


class ChurnModel(ABC):
    """Abstract base class for churn prediction models.

    All churn models must implement fit, predict, and evaluate methods.

    Parameters
    ----------
    random_state : int
        Random state for reproducibility.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    n_tuning_trials : int
        Number of trials for hyperparameter search.
    cv_folds : int
        Number of cross-validation folds.
    """

    def __init__(
        self,
        random_state: int = 42,
        tune_hyperparams: bool = False,
        n_tuning_trials: int = 30,
        cv_folds: int = 5,
    ):
        self.random_state = random_state
        self.tune_hyperparams = tune_hyperparams
        self.n_tuning_trials = n_tuning_trials
        self.cv_folds = cv_folds
        self._model: Optional[BaseEstimator] = None
        self._is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default hyperparameters."""

    @property
    @abstractmethod
    def param_distributions(self) -> dict[str, list]:
        """Return hyperparameter search space for tuning."""

    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the underlying sklearn model with given parameters."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ChurnModel:
        """Fit the model to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target.

        Returns
        -------
        self
        """
        import time

        start = time.time()

        if self.tune_hyperparams:
            self._tune_and_fit(X, y)
        else:
            self._model = self._create_model(**self.default_params)
            self._model.fit(X, y)

        self._is_fitted = True
        elapsed = time.time() - start
        logger.info("Fitted %s in %.2f seconds", self.name, elapsed)
        return self

    def _tune_and_fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        from sklearn.model_selection import RandomizedSearchCV

        base_model = self._create_model()
        search = RandomizedSearchCV(
            base_model,
            self.param_distributions,
            n_iter=self.n_tuning_trials,
            cv=self.cv_folds,
            scoring="f1",
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        self._model = search.best_estimator_
        logger.info(
            "Tuned %s: best F1=%.4f, params=%s",
            self.name,
            search.best_score_,
            search.best_params_,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn labels.

        Parameters
        ----------
        X : pd.DataFrame
            Features.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Features.

        Returns
        -------
        np.ndarray
            Predicted probabilities for the positive class.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict_proba.")
        return self._model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> ModelResult:
        """Evaluate the model on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True test labels.
        feature_names : list of str, optional
            Feature names for importance output.

        Returns
        -------
        ModelResult
            Evaluation results.
        """
        import time

        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before evaluate.")

        start = time.time()
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        result = ModelResult(
            model_name=self.name,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_proba),
            pr_auc=average_precision_score(y_test, y_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            y_pred=y_pred,
            y_proba=y_proba,
            training_time_seconds=time.time() - start,
        )

        # Feature importance
        result.feature_importance = self._get_feature_importance(feature_names)

        # Cross-validation scores
        result.cv_scores = self._compute_cv_scores(X_test, y_test)

        return result

    def _get_feature_importance(
        self, feature_names: Optional[list[str]] = None
    ) -> dict[str, float]:
        """Extract feature importance from the fitted model."""
        importances = None

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importances = np.abs(self._model.coef_[0])

        if importances is None:
            return {}

        names = feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances))

    def _compute_cv_scores(self, X: pd.DataFrame, y: pd.Series) -> list[float]:
        """Compute cross-validation F1 scores."""
        from sklearn.model_selection import cross_val_score

        try:
            scores = cross_val_score(
                self._model, X, y, cv=min(self.cv_folds, len(y)), scoring="f1", n_jobs=-1
            )
            return scores.tolist()
        except Exception as e:
            logger.warning("CV scoring failed: %s", e)
            return []

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    def get_model(self) -> BaseEstimator:
        """Return the underlying sklearn model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self._model
