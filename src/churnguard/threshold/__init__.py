"""Threshold optimization and cost-benefit analysis for churn prediction.

Provides tools to find the optimal decision threshold beyond the default 0.5:
- F1-optimal threshold
- Youden's J statistic (maximizes sensitivity + specificity - 1)
- Cost-based optimization using a business cost matrix
- Target churn rate constraint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------


@dataclass
class CostMatrix:
    """Business cost matrix for churn decisions.

    Attributes:
        tp_benefit: Revenue saved per true positive (correctly identified churner).
        fp_cost: Cost of unnecessary intervention per false positive.
        fn_cost: Revenue lost per false negative (missed churner).
        tn_benefit: Small benefit (or zero) per true negative (no intervention needed).
    """

    tp_benefit: float = 100.0
    fp_cost: float = 10.0
    fn_cost: float = 100.0
    tn_benefit: float = 0.0

    def expected_value(
        self,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
    ) -> float:
        """Compute the expected business value of a classification outcome.

        Parameters
        ----------
        tp : int
            True positives.
        fp : int
            False positives.
        fn : int
            False negatives.
        tn : int
            True negatives.

        Returns
        -------
        float
            Net business value.
        """
        return tp * self.tp_benefit - fp * self.fp_cost - fn * self.fn_cost + tn * self.tn_benefit

    def to_dict(self) -> dict[str, float]:
        """Serialize cost matrix to dict."""
        return {
            "tp_benefit": self.tp_benefit,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "tn_benefit": self.tn_benefit,
        }


# ---------------------------------------------------------------------------
# Threshold result
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Result of threshold optimization.

    Attributes:
        threshold: Optimal threshold value.
        method: Optimization method used.
        metric_value: Value of the optimized metric at the threshold.
        metrics: Dict of all metrics at the optimal threshold.
        all_thresholds: Array of evaluated thresholds.
        all_scores: Array of metric scores at each threshold.
    """

    threshold: float
    method: str
    metric_value: float
    metrics: dict[str, float]
    all_thresholds: np.ndarray | None = None
    all_scores: np.ndarray | None = None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Optimal Threshold ({self.method}): {self.threshold:.4f}",
            f"  Optimized metric value: {self.metric_value:.4f}",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "threshold": self.threshold,
            "method": self.method,
            "metric_value": self.metric_value,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Optimization functions
# ---------------------------------------------------------------------------


def _scan_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Evaluate metrics at evenly spaced thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    n_thresholds : int
        Number of threshold values to evaluate.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values.
    metrics_list : list of dict
        List of metric dicts at each threshold.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    metrics_list: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        # Handle edge case where all predictions are the same class
        len(np.unique(y_pred))
        metrics_list.append(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "threshold": float(t),
            }
        )

    return thresholds, metrics_list


def optimize_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the threshold that maximizes F1 score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    n_thresholds : int
        Number of threshold values to scan.

    Returns
    -------
    ThresholdResult
    """
    thresholds, metrics_list = _scan_thresholds(y_true, y_proba, n_thresholds)
    f1_scores = np.array([m["f1"] for m in metrics_list])

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return ThresholdResult(
        threshold=best_threshold,
        method="f1",
        metric_value=best_f1,
        metrics=metrics_list[best_idx],
        all_thresholds=thresholds,
        all_scores=f1_scores,
    )


def optimize_youden(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> ThresholdResult:
    """Find the threshold that maximizes Youden's J statistic (sensitivity + specificity - 1).

    Uses sklearn's roc_curve to find the optimal operating point.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    ThresholdResult
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr  # Youden's J = sensitivity + specificity - 1 = TPR - FPR

    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    best_j = float(j_scores[best_idx])

    # Compute metrics at the best threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(1 - fpr[best_idx]),
        "youden_j": best_j,
        "threshold": best_threshold,
    }

    return ThresholdResult(
        threshold=best_threshold,
        method="youden",
        metric_value=best_j,
        metrics=metrics,
        all_thresholds=thresholds,
        all_scores=j_scores,
    )


def optimize_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: CostMatrix | None = None,
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the threshold that maximizes expected business value.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    cost_matrix : CostMatrix, optional
        Business cost matrix. Uses defaults if None.
    n_thresholds : int
        Number of threshold values to scan.

    Returns
    -------
    ThresholdResult
    """
    if cost_matrix is None:
        cost_matrix = CostMatrix()

    thresholds, metrics_list = _scan_thresholds(y_true, y_proba, n_thresholds)
    ev_scores = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        ev_scores[i] = cost_matrix.expected_value(tp, fp, fn, tn)

    best_idx = int(np.argmax(ev_scores))
    best_threshold = float(thresholds[best_idx])
    best_ev = float(ev_scores[best_idx])

    metrics_list[best_idx]["expected_value"] = best_ev

    return ThresholdResult(
        threshold=best_threshold,
        method="cost",
        metric_value=best_ev,
        metrics=metrics_list[best_idx],
        all_thresholds=thresholds,
        all_scores=ev_scores,
    )


def optimize_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.7,
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the threshold that achieves target precision while maximizing recall.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    target_precision : float
        Minimum acceptable precision (0-1).
    n_thresholds : int
        Number of threshold values to scan.

    Returns
    -------
    ThresholdResult
    """
    thresholds, metrics_list = _scan_thresholds(y_true, y_proba, n_thresholds)

    # Filter to thresholds that meet the precision target
    valid_indices = [i for i, m in enumerate(metrics_list) if m["precision"] >= target_precision]

    if not valid_indices:
        # Fall back to the threshold with highest precision
        precisions = np.array([m["precision"] for m in metrics_list])
        best_idx = int(np.argmax(precisions))
        logger.warning(
            "No threshold achieves target precision %.2f. Best precision: %.4f at threshold %.4f",
            target_precision,
            precisions[best_idx],
            thresholds[best_idx],
        )
    else:
        # Among valid thresholds, pick the one with highest recall
        recalls = np.array([metrics_list[i]["recall"] for i in valid_indices])
        best_valid_pos = int(np.argmax(recalls))
        best_idx = valid_indices[best_valid_pos]

    best_threshold = float(thresholds[best_idx])
    recall_val = float(metrics_list[best_idx]["recall"])

    return ThresholdResult(
        threshold=best_threshold,
        method="precision_recall",
        metric_value=recall_val,
        metrics=metrics_list[best_idx],
        all_thresholds=thresholds,
        all_scores=np.array([m["recall"] for m in metrics_list]),
    )


def find_threshold_for_target_rate(
    y_proba: np.ndarray,
    target_rate: float = 0.2,
) -> ThresholdResult:
    """Find the threshold that produces a target predicted churn rate.

    Useful when business constraints require flagging only a fixed
    percentage of customers (e.g., "we can only intervene with 20%").

    Parameters
    ----------
    y_proba : array-like
        Predicted probabilities for the positive class.
    target_rate : float
        Desired fraction of customers flagged as churners (0-1).

    Returns
    -------
    ThresholdResult
    """
    if target_rate <= 0:
        # Flag nobody
        return ThresholdResult(
            threshold=1.0,
            method="target_rate",
            metric_value=0.0,
            metrics={"predicted_rate": 0.0, "threshold": 1.0},
        )

    # Sort probabilities descending and find the threshold at the target quantile
    sorted_proba = np.sort(y_proba)[::-1]
    n_flag = max(1, int(len(y_proba) * target_rate))
    threshold = float(sorted_proba[min(n_flag, len(sorted_proba)) - 1])

    # Ensure threshold doesn't go below the minimum probability
    threshold = max(threshold, float(np.min(y_proba)) + 1e-10)

    actual_rate = float(np.mean(y_proba >= threshold))

    return ThresholdResult(
        threshold=threshold,
        method="target_rate",
        metric_value=actual_rate,
        metrics={
            "predicted_rate": actual_rate,
            "target_rate": target_rate,
            "threshold": threshold,
        },
    )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "f1",
    cost_matrix: CostMatrix | None = None,
    target_precision: float = 0.7,
    target_rate: float = 0.2,
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the optimal decision threshold using the specified method.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    method : str
        Optimization method: 'f1', 'youden', 'cost', 'precision_recall'.
    cost_matrix : CostMatrix, optional
        Required for 'cost' method.
    target_precision : float
        Required for 'precision_recall' method.
    target_rate : float
        Required for 'target_rate' method (not used here, see find_threshold_for_target_rate).
    n_thresholds : int
        Number of thresholds to evaluate for scan-based methods.

    Returns
    -------
    ThresholdResult

    Raises
    ------
    ValueError
        If method is unknown.
    """
    if method == "f1":
        return optimize_f1(y_true, y_proba, n_thresholds)
    elif method == "youden":
        return optimize_youden(y_true, y_proba)
    elif method == "cost":
        return optimize_cost(y_true, y_proba, cost_matrix, n_thresholds)
    elif method == "precision_recall":
        return optimize_precision_recall(y_true, y_proba, target_precision, n_thresholds)
    else:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            f"Available: 'f1', 'youden', 'cost', 'precision_recall'"
        )


__all__ = [
    "CostMatrix",
    "ThresholdOptimizer",
    "ThresholdResult",
    "optimize_f1",
    "optimize_youden",
    "optimize_cost",
    "optimize_precision_recall",
    "find_threshold_for_target_rate",
    "optimize_threshold",
]


class ThresholdOptimizer:
    """Stateless threshold optimizer with a scikit-learn–style API.

    Parameters
    ----------
    n_thresholds : int
        Number of threshold values to scan for grid-based strategies.

    Examples
    --------
    >>> opt = ThresholdOptimizer()
    >>> result = opt.optimize(y_true, y_proba, strategy="f1")
    >>> print(result.threshold)
    """

    def __init__(self, n_thresholds: int = 100) -> None:
        self.n_thresholds = n_thresholds

    def optimize(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        strategy: str = "f1",
        cost_matrix: CostMatrix | None = None,
        target_precision: float = 0.7,
    ) -> ThresholdResult:
        """Find the optimal decision threshold.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities for the positive class.
        strategy : str
            One of ``'f1'``, ``'youden'``, ``'cost_sensitive'``, ``'precision_recall'``.
        cost_matrix : CostMatrix, optional
            Required when *strategy* is ``'cost_sensitive'``.
        target_precision : float
            Required when *strategy* is ``'precision_recall'``.

        Returns
        -------
        ThresholdResult
        """
        # Normalize strategy name (accept both "cost" and "cost_sensitive")
        method = "cost" if strategy == "cost_sensitive" else strategy
        return optimize_threshold(
            y_true,
            y_proba,
            method=method,
            cost_matrix=cost_matrix,
            target_precision=target_precision,
            n_thresholds=self.n_thresholds,
        )
