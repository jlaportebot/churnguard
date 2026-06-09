"""End-to-end churn prediction pipeline.

Orchestrates the complete workflow:
  data loading → feature engineering → model training → evaluation →
  threshold optimization → explainability → reporting

Provides both a programmatic API (``ChurnPipeline``) and CLI integration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.data import DataLoader, DataValidationError, generate_sample_data
from churnguard.evaluation import ModelEvaluator, format_results_table
from churnguard.features import FeatureEngineer
from churnguard.models import ModelRegistry
from churnguard.models.base import ModelResult
from churnguard.threshold import (
    CostMatrix,
    ThresholdResult,
    find_threshold_for_target_rate,
    optimize_threshold,
)
from churnguard.utils import ensure_dir, get_config, merge_configs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    Attributes:
        model: Model name(s) to train. Can be a single name or list.
        feature_config: Feature engineering config dict.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.
        tune_hyperparams: Whether to tune model hyperparameters.
        cv_folds: Number of cross-validation folds.
        optimize_threshold: Whether to find optimal decision threshold.
        threshold_method: Method for threshold optimization ('f1', 'youden', 'cost').
        cost_matrix: Business cost matrix (for 'cost' threshold method).
        compute_explanations: Whether to compute SHAP explanations.
        save_plots: Whether to save evaluation plots.
        output_dir: Default output directory for reports and artifacts.
        business_revenue: Revenue per customer for business impact.
        business_intervention_cost: Cost per intervention.
        business_intervention_success_rate: Probability intervention succeeds.
    """

    model: Union[str, List[str]] = "logistic"
    feature_config: Optional[Dict[str, Any]] = None
    test_size: float = 0.2
    random_state: int = 42
    tune_hyperparams: bool = False
    cv_folds: int = 5
    optimize_threshold: bool = False
    threshold_method: str = "f1"
    cost_matrix: Optional[CostMatrix] = None
    compute_explanations: bool = False
    save_plots: bool = True
    output_dir: str = "./churnguard_output"
    business_revenue: float = 100.0
    business_intervention_cost: float = 10.0
    business_intervention_success_rate: float = 0.3

    def get_model_names(self) -> List[str]:
        """Return model names as a list."""
        if isinstance(self.model, str):
            return [self.model]
        return list(self.model)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Result of running the churn prediction pipeline.

    Attributes:
        model_results: Dict of model_name → ModelResult for each trained model.
        best_model_name: Name of the best-performing model.
        threshold_result: Threshold optimization result (if computed).
        global_explanation: SHAP global explanation (if computed).
        comparison_table: DataFrame comparing all models.
        feature_engineer: Fitted FeatureEngineer (for later transforms).
        run_info: Metadata about the pipeline run.
        elapsed_seconds: Total wall-clock time.
    """

    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    threshold_result: Optional[ThresholdResult] = None
    global_explanation: Optional[Any] = None  # GlobalExplanation from explainability
    comparison_table: Optional[pd.DataFrame] = None
    feature_engineer: Optional[FeatureEngineer] = None
    run_info: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        """Comprehensive text summary."""
        lines = []

        # Run info
        if self.run_info:
            lines.append("=== Pipeline Run Info ===")
            for k, v in self.run_info.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Best model
        if self.best_model_name and self.best_model_name in self.model_results:
            best = self.model_results[self.best_model_name]
            lines.append(f"=== Best Model: {best.model_name} ===")
            lines.append(best.summary())
            lines.append("")

        # Threshold
        if self.threshold_result is not None:
            lines.append(self.threshold_result.summary())
            lines.append("")

        # Business impact
        if "business_impact" in self.run_info:
            bi = self.run_info["business_impact"]
            lines.append("=== Business Impact ===")
            for k, v in bi.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def save_report(self, path: Union[str, Path]) -> None:
        """Save pipeline results as JSON.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report: Dict[str, Any] = {
            "run_info": self.run_info,
            "best_model": self.best_model_name,
            "elapsed_seconds": self.elapsed_seconds,
        }

        # Model results
        model_summaries = {}
        for name, result in self.model_results.items():
            model_summaries[name] = result.to_dict()
        report["models"] = model_summaries

        # Threshold
        if self.threshold_result is not None:
            report["threshold"] = self.threshold_result.to_dict()

        # Business impact
        if "business_impact" in self.run_info:
            report["business_impact"] = self.run_info["business_impact"]

        # Feature importance from best model
        if self.best_model_name and self.best_model_name in self.model_results:
            best = self.model_results[self.best_model_name]
            if best.feature_importance:
                sorted_feats = sorted(
                    best.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
                report["top_features"] = dict(sorted_feats)

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Pipeline report saved to %s", path)


# ---------------------------------------------------------------------------
# Business impact computation
# ---------------------------------------------------------------------------


def compute_business_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    revenue_per_customer: float = 100.0,
    intervention_cost: float = 10.0,
    intervention_success_rate: float = 0.3,
) -> Dict[str, float]:
    """Compute the business impact of a churn prediction model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted churn probabilities.
    threshold : float
        Decision threshold.
    revenue_per_customer : float
        Revenue lost per churned customer.
    intervention_cost : float
        Cost of intervening on a customer.
    intervention_success_rate : float
        Probability that an intervention prevents churn.

    Returns
    -------
    dict
        Business impact metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    n_churners = int(np.sum(y_true))
    n_flagged = int(np.sum(y_pred))

    # Revenue saved: true positives × intervention success × revenue
    revenue_saved = tp * intervention_success_rate * revenue_per_customer

    # Intervention cost: all flagged × cost
    total_intervention_cost = n_flagged * intervention_cost

    # Revenue lost: false negatives × revenue (missed churners)
    revenue_lost = fn * revenue_per_customer

    # Net value
    net_value = revenue_saved - total_intervention_cost - revenue_lost

    # ROI
    total_cost = total_intervention_cost + revenue_lost
    roi = (net_value / total_cost * 100) if total_cost > 0 else 0.0

    # Baseline: no model → all churners leave, no intervention cost
    baseline_revenue_lost = n_churners * revenue_per_customer

    # Value vs baseline
    value_vs_baseline = (
        (revenue_saved - total_intervention_cost) / baseline_revenue_lost * 100
        if baseline_revenue_lost > 0
        else 0.0
    )

    return {
        "revenue_saved": round(revenue_saved, 2),
        "intervention_cost": round(total_intervention_cost, 2),
        "revenue_lost": round(revenue_lost, 2),
        "net_value": round(net_value, 2),
        "roi_percent": round(roi, 2),
        "value_vs_baseline_percent": round(value_vs_baseline, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "n_churners": n_churners,
        "n_flagged": n_flagged,
    }


# ---------------------------------------------------------------------------
# ChurnPipeline
# ---------------------------------------------------------------------------


class ChurnPipeline:
    """End-to-end churn prediction pipeline.

    Orchestrates data loading, feature engineering, model training,
    evaluation, threshold optimization, explainability, and reporting.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if None.

    Examples
    --------
    >>> pipeline = ChurnPipeline(config=PipelineConfig(model="random_forest"))
    >>> result = pipeline.run("data/churn_data.csv", target="churn")
    >>> print(result.summary())
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._registry: Optional[ModelRegistry] = None
        self._engineer: Optional[FeatureEngineer] = None
        self._best_model: Optional[Any] = None
        self._is_fitted = False

    def run(
        self,
        data_path: str,
        target: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full churn prediction pipeline.

        Parameters
        ----------
        data_path : str
            Path to the CSV data file.
        target : str, optional
            Target column name. Auto-detected if None.
        output_dir : str, optional
            Override output directory from config.

        Returns
        -------
        PipelineResult
        """
        start_time = time.time()
        cfg = self.config
        timestamp = datetime.now(timezone.utc).isoformat()

        out_dir = Path(output_dir or cfg.output_dir)
        ensure_dir(out_dir)

        # Step 1: Load data
        logger.info("Loading data from %s", data_path)
        loader = DataLoader(data_path, target_column=target, random_state=cfg.random_state)
        X_train, X_test, y_train, y_test = loader.split(test_size=cfg.test_size)
        logger.info(
            "Loaded %d rows, target='%s', churn_rate=%.1f%%",
            len(X_train) + len(X_test),
            loader.target_name,
            y_train.mean() * 100,
        )

        # Step 2: Feature engineering
        logger.info("Engineering features...")
        feat_config = cfg.feature_config or get_config().get("features", {})
        engineer = FeatureEngineer(
            numeric_impute_strategy=feat_config.get("numeric_impute_strategy", "median"),
            scaling=feat_config.get("scaling", "standard"),
            categorical_max_cardinality=feat_config.get("categorical_max_cardinality", 20),
            generate_interactions=feat_config.get("generate_interactions", False),
        )
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)
        self._engineer = engineer
        logger.info("Engineered %d features", X_train_tf.shape[1])

        # Step 3: Train models
        logger.info("Training models: %s", cfg.get_model_names())
        registry = ModelRegistry(
            models=cfg.get_model_names(),
            random_state=cfg.random_state,
            tune_hyperparams=cfg.tune_hyperparams,
            cv_folds=cfg.cv_folds,
        )
        self._registry = registry
        # Step 3: Train and evaluate models
        model_results: Dict[str, ModelResult] = {}
        trained_models: Dict[str, Any] = {}
        for model_name in registry.available_models:
            try:
                result = registry.train_and_evaluate(
                    model_name,
                    X_train_tf,
                    X_test_tf,
                    y_train,
                    y_test,
                    feature_names=list(X_train_tf.columns),
                )
                model_results[model_name] = result
                trained_models[model_name] = registry.get_model(model_name)
                logger.info(
                    "Trained %s: F1=%.4f, AUC=%.4f",
                    model_name,
                    result.f1,
                    result.roc_auc,
                )
            except Exception as e:
                logger.error("Failed to train %s: %s", model_name, e)

        if not model_results:
            raise RuntimeError("All models failed to train!")

        # Step 4: Select best model
        primary_metric = get_config().get("evaluation", {}).get("primary_metric", "f1")
        best = registry.get_best(model_results, metric=primary_metric)
        # Find the registry key corresponding to the best model
        best_key = None
        for key, result in model_results.items():
            if result is best:
                best_key = key
                break
        if best_key is None:
            best_key = registry.available_models[0]
        self._best_model = trained_models[best_key]
        self._is_fitted = True

        best_name = best.model_name
        logger.info("Best model: %s (F1=%.4f)", best_name, best.f1)

        # Step 5: Threshold optimization
        threshold_result = None
        if cfg.optimize_threshold and best.y_proba is not None:
            threshold_result = optimize_threshold(
                y_test.values,
                best.y_proba,
                method=cfg.threshold_method,
                cost_matrix=cfg.cost_matrix,
            )
            logger.info(
                "Optimal threshold (%s): %.4f",
                cfg.threshold_method,
                threshold_result.threshold,
            )

        # Step 6: Business impact
        effective_threshold = threshold_result.threshold if threshold_result else 0.5
        if best.y_proba is not None:
            business_impact = compute_business_impact(
                y_test.values,
                best.y_proba,
                threshold=effective_threshold,
                revenue_per_customer=cfg.business_revenue,
                intervention_cost=cfg.business_intervention_cost,
                intervention_success_rate=cfg.business_intervention_success_rate,
            )
        else:
            business_impact = {}

        # Step 7: Explainability
        global_explanation = None
        if cfg.compute_explanations:
            try:
                from churnguard.explainability import ChurnExplainer

                inner_model = (
                    self._best_model._model
                    if hasattr(self._best_model, "_model")
                    else self._best_model
                )
                explainer = ChurnExplainer(model=inner_model)
                explainer.fit(X_train_tf, feature_names=list(X_train_tf.columns))
                global_explanation = explainer.explain_global(
                    X_test_tf.iloc[:100] if len(X_test_tf) > 100 else X_test_tf
                )
                logger.info("Computed SHAP explanations")
            except ImportError:
                logger.warning("shap not installed — skipping explanations")
            except Exception as e:
                logger.warning("Explainability failed: %s", e)

        # Step 8: Evaluation outputs
        evaluator = ModelEvaluator(
            output_dir=out_dir,
            save_plots=cfg.save_plots,
            save_json=True,
        )
        for name, result in model_results.items():
            evaluator.evaluate_model(result, y_test)

        comparison_table = evaluator.compare_models(model_results, y_test)
        comparison_table.to_csv(out_dir / "model_comparison.csv")

        # Build result
        elapsed = time.time() - start_time

        pipeline_result = PipelineResult(
            model_results=model_results,
            best_model_name=best_name,
            threshold_result=threshold_result,
            global_explanation=global_explanation,
            comparison_table=comparison_table,
            feature_engineer=engineer,
            run_info={
                "timestamp": timestamp,
                "data_path": data_path,
                "target": loader.target_name,
                "n_rows": len(X_train) + len(X_test),
                "n_features": X_train_tf.shape[1],
                "churn_rate": float(y_train.mean()),
                "best_model": best_name,
                "threshold": effective_threshold,
                "models_trained": list(model_results.keys()),
                "elapsed_seconds": round(elapsed, 2),
                "business_impact": business_impact,
            },
            elapsed_seconds=elapsed,
        )

        # Save JSON report
        pipeline_result.save_report(out_dir / "pipeline_report.json")

        logger.info("Pipeline complete in %.1fs. Best: %s", elapsed, best_name)
        return pipeline_result

    def predict(self, data_path: str) -> pd.DataFrame:
        """Make predictions on new data using the fitted pipeline.

        Parameters
        ----------
        data_path : str
            Path to the CSV file with new customer data.

        Returns
        -------
        pd.DataFrame
            Predictions with churn_probability and churn_label columns.
        """
        if not self._is_fitted or self._best_model is None:
            raise RuntimeError("Pipeline must be run before making predictions.")
        if self._engineer is None:
            raise RuntimeError("Feature engineer not available.")

        df = pd.read_csv(data_path)
        X_tf = self._engineer.transform(df)

        proba = self._best_model.predict_proba(X_tf)
        labels = (proba >= 0.5).astype(int)

        result = df.copy()
        result["churn_probability"] = proba
        result["churn_label"] = labels
        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted pipeline to disk.

        Parameters
        ----------
        path : str or Path
            Output path for the saved pipeline.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "config": self.config,
            "engineer": self._engineer,
            "best_model_name": self.config.get_model_names() if not self._is_fitted else None,
        }

        if self._registry is not None:
            artifact["registry"] = self._registry

        joblib.dump(artifact, path)
        logger.info("Pipeline saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ChurnPipeline:
        """Load a saved pipeline from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved pipeline file.

        Returns
        -------
        ChurnPipeline
        """
        import joblib

        artifact = joblib.load(path)
        pipeline = cls(config=artifact.get("config", PipelineConfig()))
        pipeline._engineer = artifact.get("engineer")
        pipeline._registry = artifact.get("registry")
        pipeline._is_fitted = True

        return pipeline


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_pipeline(
    data_path: str,
    target: Optional[str] = None,
    model: Union[str, List[str]] = "logistic",
    output_dir: str = "./churnguard_output",
    optimize_threshold: bool = False,
    threshold_method: str = "f1",
    compute_explanations: bool = False,
    random_state: int = 42,
    test_size: float = 0.2,
    tune_hyperparams: bool = False,
) -> PipelineResult:
    """Run the churn prediction pipeline with a simple function call.

    Parameters
    ----------
    data_path : str
        Path to the CSV data file.
    target : str, optional
        Target column name.
    model : str or list of str
        Model name(s) to train.
    output_dir : str
        Output directory.
    optimize_threshold : bool
        Whether to optimize the decision threshold.
    threshold_method : str
        Threshold optimization method.
    compute_explanations : bool
        Whether to compute SHAP explanations.
    random_state : int
        Random seed.
    test_size : float
        Test set fraction.
    tune_hyperparams : bool
        Whether to tune model hyperparameters.

    Returns
    -------
    PipelineResult
    """
    config = PipelineConfig(
        model=model,
        optimize_threshold=optimize_threshold,
        threshold_method=threshold_method,
        compute_explanations=compute_explanations,
        random_state=random_state,
        test_size=test_size,
        tune_hyperparams=tune_hyperparams,
        output_dir=output_dir,
    )
    pipeline = ChurnPipeline(config=config)
    return pipeline.run(data_path, target=target)


__all__ = [
    "ChurnPipeline",
    "PipelineConfig",
    "PipelineResult",
    "compute_business_impact",
    "run_pipeline",
]
