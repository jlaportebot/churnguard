"""Model performance monitoring for churn prediction.

Tracks model metrics over time, detects degradation, and provides
historical analysis of model performance trends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSnapshot:
    """A snapshot of model performance at a point in time.

    Attributes:
        timestamp: When the measurement was taken (ISO format).
        model_name: Name of the model.
        n_samples: Number of samples evaluated.
        accuracy: Accuracy score.
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the PR curve.
        churn_rate: Observed churn rate in the evaluated data.
        threshold: Decision threshold used.
        metadata: Additional key-value metadata.
    """

    timestamp: str
    model_name: str
    n_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    churn_rate: float = 0.0
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "churn_rate": round(self.churn_rate, 6),
            "threshold": self.threshold,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.timestamp}] {self.model_name}: "
            f"F1={self.f1:.4f}, AUC={self.roc_auc:.4f}, "
            f"Recall={self.recall:.4f}, Prec={self.precision:.4f} "
            f"(n={self.n_samples})"
        )


@dataclass
class PerformanceAlert:
    """Alert triggered by performance degradation.

    Attributes:
        timestamp: When the alert was generated.
        metric_name: Name of the degraded metric.
        current_value: Current metric value.
        baseline_value: Baseline (expected) metric value.
        degradation_pct: Percentage degradation from baseline.
        severity: Alert severity ('warning' or 'critical').
        model_name: Name of the model.
    """

    timestamp: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str
    model_name: str

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.severity.upper()}] {self.model_name}: "
            f"{self.metric_name} degraded by {self.degradation_pct:.1f}% "
            f"({self.baseline_value:.4f} → {self.current_value:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 6),
            "baseline_value": round(self.baseline_value, 6),
            "degradation_pct": round(self.degradation_pct, 2),
            "severity": self.severity,
            "model_name": self.model_name,
        }


class MetricHistory:
    """Track the history of a single metric over time.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g. 'f1', 'roc_auc').
    """

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self._timestamps: list[str] = []
        self._values: list[float] = []

    def add(self, timestamp: str, value: float) -> None:
        """Add a new measurement.

        Parameters
        ----------
        timestamp : str
            ISO timestamp.
        value : float
            Metric value.
        """
        self._timestamps.append(timestamp)
        self._values.append(value)

    @property
    def values(self) -> np.ndarray:
        """Return all values as a numpy array."""
        return np.array(self._values)

    @property
    def timestamps(self) -> list[str]:
        """Return all timestamps."""
        return self._timestamps

    def latest(self) -> float | None:
        """Return the most recent value."""
        return self._values[-1] if self._values else None

    def baseline(self) -> float | None:
        """Return the first (baseline) value."""
        return self._values[0] if self._values else None

    def mean(self) -> float:
        """Return mean of all values."""
        return float(np.mean(self._values)) if self._values else 0.0

    def std(self) -> float:
        """Return standard deviation of all values."""
        return float(np.std(self._values)) if len(self._values) > 1 else 0.0

    def trend(self) -> str:
        """Return the trend direction: 'improving', 'stable', or 'degrading'.

        Computed using simple linear regression slope.
        """
        if len(self._values) < 3:
            return "stable"

        x = np.arange(len(self._values))
        y = np.array(self._values)
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope relative to the mean
        mean_val = self.mean()
        if mean_val == 0:
            return "stable"

        relative_slope = slope / mean_val

        if relative_slope > 0.005:
            return "improving"
        elif relative_slope < -0.005:
            return "degrading"
        else:
            return "stable"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": self._timestamps,
                self.metric_name: self._values,
            }
        )

    def __len__(self) -> int:
        return len(self._values)


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Monitor model performance over time and detect degradation.

    Parameters
    ----------
    model_name : str
        Name of the model being monitored.
    warning_threshold_pct : float
        Percentage degradation for warning alerts (default 5%).
    critical_threshold_pct : float
        Percentage degradation for critical alerts (default 15%).
    metrics_to_track : list of str, optional
        Metric names to track. Defaults to standard classification metrics.
    """

    DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    def __init__(
        self,
        model_name: str = "churn_model",
        warning_threshold_pct: float = 5.0,
        critical_threshold_pct: float = 15.0,
        metrics_to_track: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self._metrics_to_track = metrics_to_track or self.DEFAULT_METRICS
        self._history: dict[str, MetricHistory] = {
            m: MetricHistory(m) for m in self._metrics_to_track
        }
        self._snapshots: list[PerformanceSnapshot] = []
        self._alerts: list[PerformanceAlert] = []

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        threshold: float = 0.5,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PerformanceSnapshot:
        """Evaluate model performance and record a snapshot.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        y_proba : np.ndarray, optional
            Predicted probabilities for the positive class.
        threshold : float
            Decision threshold used.
        timestamp : str, optional
            ISO timestamp. Defaults to current time.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        PerformanceSnapshot
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        n_samples = len(y_true)
        churn_rate = float(np.mean(y_true))

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            model_name=self.model_name,
            n_samples=n_samples,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            churn_rate=churn_rate,
            threshold=threshold,
            metadata=metadata or {},
        )

        # Record in history
        for metric_name, value in metrics.items():
            if metric_name in self._history:
                self._history[metric_name].add(timestamp, value)

        self._snapshots.append(snapshot)

        # Check for degradation alerts
        new_alerts = self._check_degradation(snapshot, timestamp)
        self._alerts.extend(new_alerts)

        return snapshot

    def _check_degradation(
        self, snapshot: PerformanceSnapshot, timestamp: str
    ) -> list[PerformanceAlert]:
        """Check if any metrics have degraded relative to baseline.

        Parameters
        ----------
        snapshot : PerformanceSnapshot
            Current performance snapshot.
        timestamp : str
            Current timestamp.

        Returns
        -------
        list of PerformanceAlert
        """
        alerts = []

        # Only check after we have a baseline (at least 2 snapshots)
        if len(self._snapshots) < 2:
            return alerts

        baseline = self._snapshots[0]

        for metric_name in self._metrics_to_track:
            current_val = getattr(snapshot, metric_name, None)
            baseline_val = getattr(baseline, metric_name, None)

            if current_val is None or baseline_val is None or baseline_val == 0:
                continue

            degradation_pct = (baseline_val - current_val) / baseline_val * 100

            if degradation_pct >= self.critical_threshold_pct:
                alerts.append(
                    PerformanceAlert(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        current_value=current_val,
                        baseline_value=baseline_val,
                        degradation_pct=degradation_pct,
                        severity="critical",
                        model_name=self.model_name,
                    )
                )
            elif degradation_pct >= self.warning_threshold_pct:
                alerts.append(
                    PerformanceAlert(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        current_value=current_val,
                        baseline_value=baseline_val,
                        degradation_pct=degradation_pct,
                        severity="warning",
                        model_name=self.model_name,
                    )
                )

        return alerts

    @property
    def alerts(self) -> list[PerformanceAlert]:
        """All alerts generated so far."""
        return self._alerts

    @property
    def snapshots(self) -> list[PerformanceSnapshot]:
        """All recorded snapshots."""
        return self._snapshots

    @property
    def history(self) -> dict[str, MetricHistory]:
        """Metric history objects."""
        return self._history

    def get_trends(self) -> dict[str, str]:
        """Get the current trend for each tracked metric.

        Returns
        -------
        dict
            Metric name → trend direction ('improving', 'stable', 'degrading').
        """
        return {name: hist.trend() for name, hist in self._history.items()}

    def summary(self) -> str:
        """Human-readable summary of monitoring state."""
        lines = [
            f"=== Performance Monitor: {self.model_name} ===",
            f"Snapshots: {len(self._snapshots)}",
            f"Alerts: {len(self._alerts)}",
            "",
            "Current trends:",
        ]
        for metric, trend in self.get_trends().items():
            hist = self._history[metric]
            latest = hist.latest()
            baseline = hist.baseline()
            lines.append(
                f"  {metric}: {trend} (baseline={baseline:.4f}, current={latest:.4f})"
                if latest is not None and baseline is not None
                else f"  {metric}: {trend} (no data)"
            )

        if self._alerts:
            lines.append("")
            lines.append("Recent alerts:")
            for alert in self._alerts[-5:]:
                lines.append(f"  {alert.summary()}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all snapshots to a pandas DataFrame."""
        if not self._snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self._snapshots])

    def alerts_to_dataframe(self) -> pd.DataFrame:
        """Convert all alerts to a pandas DataFrame."""
        if not self._alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._alerts])
