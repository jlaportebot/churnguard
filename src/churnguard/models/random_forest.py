"""Random Forest churn model."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from churnguard.models.base import ChurnModel


class RandomForestChurnModel(ChurnModel):
    """Random Forest model for churn prediction.

    Ensemble of decision trees with bagging. Good baseline that
    handles mixed feature types and non-linear relationships.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree.
    min_samples_split : int
        Minimum samples to split a node.
    min_samples_leaf : int
        Minimum samples in a leaf node.
    random_state : int
        Random state for reproducibility.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        }

    @property
    def param_distributions(self) -> dict[str, list]:
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "class_weight": ["balanced", "balanced_subsample", None],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
        }

    def _create_model(self, **params) -> RandomForestClassifier:
        defaults = self.default_params
        defaults.update(params)
        return RandomForestClassifier(**defaults)
