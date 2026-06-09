"""Model evaluation and visualization module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churnguard.models.base import ModelResult

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator with rich output and visualization.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory to save evaluation outputs.
    save_plots : bool
        Whether to save plots to disk.
    save_json : bool
        Whether to save results as JSON.
    plot_format : str
        Plot file format: 'png', 'svg', or 'pdf'.
    dpi : int
        DPI for saved plots.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        save_plots: bool = True,
        save_json: bool = True,
        plot_format: str = "png",
        dpi: int = 150,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_plots = save_plots
        self.save_json = save_json
        self.plot_format = plot_format
        self.dpi = dpi

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model_result: ModelResult,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Produce a comprehensive evaluation report for a single model.

        Parameters
        ----------
        model_result : ModelResult
            The model result to evaluate.
        y_test : pd.Series
            True test labels.

        Returns
        -------
        dict
            Comprehensive evaluation report.
        """
        report = {
            "model_name": model_result.model_name,
            "metrics": {
                "accuracy": model_result.accuracy,
                "precision": model_result.precision,
                "recall": model_result.recall,
                "f1": model_result.f1,
                "roc_auc": model_result.roc_auc,
                "pr_auc": model_result.pr_auc,
            },
            "confusion_matrix": model_result.confusion_matrix.tolist()
            if model_result.confusion_matrix is not None
            else None,
            "classification_report": None,
            "feature_importance": model_result.feature_importance,
            "training_time_seconds": model_result.training_time_seconds,
        }

        # Classification report
        if model_result.y_pred is not None:
            report["classification_report"] = classification_report(
                y_test, model_result.y_pred, output_dict=True, zero_division=0
            )

        # Churn risk distribution
        if model_result.y_proba is not None:
            proba = model_result.y_proba
            report["risk_distribution"] = {
                "very_high_75_100": int((proba >= 0.75).sum()),
                "high_50_75": int(((proba >= 0.50) & (proba < 0.75)).sum()),
                "medium_25_50": int(((proba >= 0.25) & (proba < 0.50)).sum()),
                "low_0_25": int((proba < 0.25).sum()),
            }

        # Save JSON
        if self.output_dir and self.save_json:
            json_path = (
                self.output_dir / f"{model_result.model_name.replace(' ', '_').lower()}_report.json"
            )
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Saved evaluation report to %s", json_path)

        # Generate plots
        if self.save_plots and self.output_dir:
            self._plot_confusion_matrix(model_result)
            self._plot_roc_curve(model_result, y_test)
            self._plot_pr_curve(model_result, y_test)
            self._plot_feature_importance(model_result)
            self._plot_risk_distribution(model_result)

        return report

    def compare_models(
        self,
        results: dict[str, ModelResult],
        y_test: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compare multiple models and generate comparison plots.

        Parameters
        ----------
        results : dict[str, ModelResult]
            Model name → result mapping.
        y_test : pd.Series, optional
            True test labels for per-model evaluation.

        Returns
        -------
        pd.DataFrame
            Comparison table.
        """
        rows = []
        for name, result in results.items():
            rows.append(result.to_dict())

        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        if self.save_plots and self.output_dir:
            self._plot_model_comparison(results)

        return df

    def _plot_confusion_matrix(self, result: ModelResult) -> Optional[Path]:
        """Plot confusion matrix."""
        if result.confusion_matrix is None:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed. Skipping plot.")
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            result.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title(f"Confusion Matrix — {result.model_name}")

        path = (
            self.output_dir / f"cm_{result.model_name.replace(' ', '_').lower()}.{self.plot_format}"
        )
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_roc_curve(self, result: ModelResult, y_test: pd.Series) -> Optional[Path]:
        """Plot ROC curve."""
        if result.y_proba is None:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fpr, tpr, _ = roc_curve(y_test, result.y_proba)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"{result.model_name} (AUC={result.roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = (
            self.output_dir
            / f"roc_{result.model_name.replace(' ', '_').lower()}.{self.plot_format}"
        )
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_pr_curve(self, result: ModelResult, y_test: pd.Series) -> Optional[Path]:
        """Plot Precision-Recall curve."""
        if result.y_proba is None:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, result.y_proba)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall_vals, precision_vals, label=f"{result.model_name} (AP={result.pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = (
            self.output_dir / f"pr_{result.model_name.replace(' ', '_').lower()}.{self.plot_format}"
        )
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_feature_importance(self, result: ModelResult) -> Optional[Path]:
        """Plot top feature importances."""
        if not result.feature_importance:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        sorted_feats = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[
            :15
        ]
        names, values = zip(*sorted_feats) if sorted_feats else ([], [])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(names)), values, align="center")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top Features — {result.model_name}")
        ax.invert_yaxis()

        path = (
            self.output_dir
            / f"importance_{result.model_name.replace(' ', '_').lower()}.{self.plot_format}"
        )
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_risk_distribution(self, result: ModelResult) -> Optional[Path]:
        """Plot churn risk distribution."""
        if result.y_proba is None:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(result.y_proba, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0.5, color="red", linestyle="--", label="Decision boundary")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Churn Risk Distribution — {result.model_name}")
        ax.legend()

        path = (
            self.output_dir
            / f"risk_dist_{result.model_name.replace(' ', '_').lower()}.{self.plot_format}"
        )
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_model_comparison(self, results: dict[str, ModelResult]) -> Optional[Path]:
        """Plot model comparison bar chart."""
        if not results:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        names = list(results.keys())
        values = {m: [getattr(results[n], m) for n in names] for m in metrics}

        x = np.arange(len(names))
        width = 0.15

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, values[metric], width, label=metric.upper())

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(names)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        path = self.output_dir / f"model_comparison.{self.plot_format}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path


def format_results_table(results: dict[str, ModelResult]) -> str:
    """Format model results as a human-readable text table.

    Parameters
    ----------
    results : dict[str, ModelResult]
        Model name → result mapping.

    Returns
    -------
    str
        Formatted text table.

    Raises
    ------
    ValueError
        If results dict is empty.
    """
    if not results:
        raise ValueError("No results to format.")

    header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'PR-AUC':>6} {'Time':>6}"
    sep = "-" * len(header)
    lines = [header, sep]

    for result in sorted(results.values(), key=lambda r: r.f1, reverse=True):
        line = (
            f"{result.model_name:<22} "
            f"{result.accuracy:>6.3f} "
            f"{result.precision:>6.3f} "
            f"{result.recall:>6.3f} "
            f"{result.f1:>6.3f} "
            f"{result.roc_auc:>6.3f} "
            f"{result.pr_auc:>6.3f} "
            f"{result.training_time_seconds:>5.1f}s"
        )
        lines.append(line)

    return "\n".join(lines)
