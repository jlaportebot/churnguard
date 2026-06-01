"""Logistic Regression churn model."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from churnguard.models.base import ChurnModel


class LogisticChurnModel(ChurnModel):
    """Logistic Regression model for churn prediction.

    Provides a fast, interpretable baseline with L2 regularization.
    Supports hyperparameter tuning over C, penalty, and solver.

    Parameters
    ----------
    C : float
        Inverse of regularization strength.
    max_iter : int
        Maximum number of iterations.
    solver : str
        Optimization algorithm.
    random_state : int
        Random state for reproducibility.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
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
        self.C = C
        self.max_iter = max_iter
        self.solver = solver

    @property
    def name(self) -> str:
        return "Logistic Regression"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "random_state": self.random_state,
            "class_weight": "balanced",
        }

    @property
    def param_distributions(self) -> dict[str, list]:
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [500, 1000, 2000],
            "class_weight": ["balanced", None],
        }

    def _create_model(self, **params) -> LogisticRegression:
        defaults = self.default_params
        defaults.update(params)
        return LogisticRegression(**defaults)
