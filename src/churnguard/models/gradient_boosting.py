"""Gradient Boosting churn model."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier

from churnguard.models.base import ChurnModel


class GradientBoostingChurnModel(ChurnModel):
    """Gradient Boosting model for churn prediction.

    Sequential ensemble that builds trees to correct errors of previous trees.
    Often achieves the best performance but is slower to train.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    learning_rate : float
        Learning rate (shrinks contribution of each tree).
    max_depth : int
        Maximum depth of individual trees.
    subsample : float
        Fraction of samples used for fitting each tree.
    random_state : int
        Random state for reproducibility.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        random_state: int = 42,
        tune_hyperparams: bool = False,
        n_tuning_trials: int = 30,
        cv_folds: int = 5,
    ):
        super().__init__(
            random_state=random_state,
            tune_hyperparams=tune_hyperparams,
            n_tuning_trials=n_tuning_trials,
            cv_folds=cv_folds,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

    @property
    def name(self) -> str:
        return "Gradient Boosting"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "random_state": self.random_state,
        }

    @property
    def param_distributions(self) -> dict[str, list]:
        return {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 7, 9],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        }

    def _create_model(self, **params) -> GradientBoostingClassifier:
        defaults = self.default_params
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)
