"""Model registry for discovering and comparing churn models."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from churnguard.models.base import ChurnModel, ModelResult
from churnguard.models.gradient_boosting import GradientBoostingChurnModel
from churnguard.models.logistic import LogisticChurnModel
from churnguard.models.random_forest import RandomForestChurnModel

logger = logging.getLogger(__name__)

# Built-in models available by default
BUILTIN_MODELS: dict[str, type[ChurnModel]] = {
    "logistic": LogisticChurnModel,
    "random_forest": RandomForestChurnModel,
    "gradient_boosting": GradientBoostingChurnModel,
}


class ModelRegistry:
    """Registry for discovering, instantiating, and comparing churn models.

    Supports built-in models and models registered via entry points.

    Parameters
    ----------
    models : list of str, optional
        Names of models to include. If None, all built-in models are used.
    random_state : int
        Random state for reproducibility.
    tune_hyperparams : bool
        Whether to tune hyperparameters for all models.
    cv_folds : int
        Number of cross-validation folds.
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        random_state: int = 42,
        tune_hyperparams: bool = False,
        cv_folds: int = 5,
    ):
        self.random_state = random_state
        self.tune_hyperparams = tune_hyperparams
        self.cv_folds = cv_folds
        self._model_instances: dict[str, ChurnModel] = {}
        self._results: dict[str, ModelResult] = {}

        # Initialize model instances
        model_names = models or list(BUILTIN_MODELS.keys())
        for name in model_names:
            if name in BUILTIN_MODELS:
                self._model_instances[name] = BUILTIN_MODELS[name](
                    random_state=random_state,
                    tune_hyperparams=tune_hyperparams,
                    cv_folds=cv_folds,
                )
            else:
                logger.warning("Unknown model: '%s'. Skipping.", name)

    @property
    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self._model_instances.keys())

    def get_model(self, name: str) -> ChurnModel:
        """Get a model instance by name.

        Parameters
        ----------
        name : str
            Model name (e.g., 'logistic', 'random_forest').

        Returns
        -------
        ChurnModel
            The model instance.

        Raises
        ------
        KeyError
            If the model name is not found.
        """
        if name not in self._model_instances:
            raise KeyError(f"Model '{name}' not found. Available: {self.available_models}")
        return self._model_instances[name]

    def train_model(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> ChurnModel:
        """Train a single model by name.

        Parameters
        ----------
        name : str
            Model name.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.

        Returns
        -------
        ChurnModel
            The fitted model.
        """
        model = self.get_model(name)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self,
        name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> ModelResult:
        """Evaluate a single model.

        Parameters
        ----------
        name : str
            Model name.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True test labels.
        feature_names : list of str, optional
            Feature names for importance.

        Returns
        -------
        ModelResult
        """
        model = self.get_model(name)
        if not model.is_fitted:
            raise RuntimeError(f"Model '{name}' must be trained before evaluation.")
        result = model.evaluate(X_test, y_test, feature_names=feature_names)
        self._results[name] = result
        return result

    def train_and_evaluate(
        self,
        name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> ModelResult:
        """Train and evaluate a single model.

        Parameters
        ----------
        name : str
            Model name.
        X_train, X_test : pd.DataFrame
            Feature matrices.
        y_train, y_test : pd.Series
            Target vectors.
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        ModelResult
        """
        self.train_model(name, X_train, y_train)
        return self.evaluate_model(name, X_test, y_test, feature_names=feature_names)

    def compare_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, ModelResult]:
        """Train and evaluate all registered models.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Feature matrices.
        y_train, y_test : pd.Series
            Target vectors.
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        dict[str, ModelResult]
            Model name → results mapping.
        """
        results = {}
        for name in self.available_models:
            try:
                result = self.train_and_evaluate(
                    name, X_train, X_test, y_train, y_test, feature_names=feature_names
                )
                results[name] = result
                logger.info("%s: F1=%.4f, ROC-AUC=%.4f", name, result.f1, result.roc_auc)
            except Exception as e:
                logger.error("Failed to train/evaluate '%s': %s", name, e)

        self._results.update(results)
        return results

    def get_best(
        self,
        results: Optional[dict[str, ModelResult]] = None,
        metric: str = "f1",
    ) -> ModelResult:
        """Get the best model result by a given metric.

        Parameters
        ----------
        results : dict, optional
            Results to compare. If None, uses stored results.
        metric : str
            Metric to rank by: 'f1', 'roc_auc', 'accuracy', 'precision', 'recall'.

        Returns
        -------
        ModelResult
            The best result.
        """
        all_results = results or self._results
        if not all_results:
            raise ValueError("No results available for comparison.")

        return max(all_results.values(), key=lambda r: getattr(r, metric))

    def comparison_table(self, results: Optional[dict[str, ModelResult]] = None) -> pd.DataFrame:
        """Generate a comparison table of all model results.

        Parameters
        ----------
        results : dict, optional
            Results to compare. If None, uses stored results.

        Returns
        -------
        pd.DataFrame
            Comparison table sorted by F1 score.
        """
        all_results = results or self._results
        if not all_results:
            raise ValueError("No results available.")

        rows = []
        for result in all_results.values():
            rows.append(result.to_dict())

        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1-based ranking
        return df

    @property
    def results(self) -> dict[str, ModelResult]:
        """Return all stored results."""
        return self._results
