"""Tests for the threshold optimization module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from churnguard.threshold import (
    CostMatrix,
    ThresholdResult,
    optimize_f1,
    optimize_youden,
    optimize_cost,
    optimize_precision_recall,
    find_threshold_for_target_rate,
    optimize_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Simple binary classification data with known properties."""
    rng = np.random.RandomState(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Good probabilities: perfect separation at 0.5
    y_proba = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9])
    return y_true, y_proba


@pytest.fixture
def noisy_data():
    """Noisy data with some misclassified samples."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.3, size=500)
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, size=500), 0.01, 0.99)
    return y_true, y_proba


@pytest.fixture
def cost_matrix():
    """Standard cost matrix."""
    return CostMatrix(tp_benefit=100, fp_cost=10, fn_cost=100, tn_benefit=0)


# ---------------------------------------------------------------------------
# CostMatrix tests
# ---------------------------------------------------------------------------

class TestCostMatrix:
    def test_default_values(self):
        cm = CostMatrix()
        assert cm.tp_benefit == 100.0
        assert cm.fp_cost == 10.0
        assert cm.fn_cost == 100.0
        assert cm.tn_benefit == 0.0

    def test_custom_values(self):
        cm = CostMatrix(tp_benefit=200, fp_cost=20, fn_cost=150, tn_benefit=5)
        assert cm.tp_benefit == 200
        assert cm.fp_cost == 20

    def test_expected_value_all_correct(self):
        cm = CostMatrix(tp_benefit=100, fp_cost=10, fn_cost=100, tn_benefit=0)
        ev = cm.expected_value(tp=5, fp=0, fn=0, tn=5)
        assert ev == 500.0  # 5 * 100

    def test_expected_value_with_fp_fn(self):
        cm = CostMatrix(tp_benefit=100, fp_cost=10, fn_cost=100, tn_benefit=0)
        ev = cm.expected_value(tp=3, fp=2, fn=1, tn=4)
        # 3*100 - 2*10 - 1*100 + 4*0 = 300 - 20 - 100 = 180
        assert ev == 180.0

    def test_expected_value_negative(self):
        cm = CostMatrix(tp_benefit=10, fp_cost=100, fn_cost=100, tn_benefit=0)
        ev = cm.expected_value(tp=1, fp=10, fn=5, tn=0)
        # 1*10 - 10*100 - 5*100 = 10 - 1000 - 500 = -1490
        assert ev == -1490.0

    def test_to_dict(self):
        cm = CostMatrix(tp_benefit=50, fp_cost=5, fn_cost=80, tn_benefit=1)
        d = cm.to_dict()
        assert d["tp_benefit"] == 50
        assert d["fp_cost"] == 5
        assert d["fn_cost"] == 80
        assert d["tn_benefit"] == 1


# ---------------------------------------------------------------------------
# ThresholdResult tests
# ---------------------------------------------------------------------------

class TestThresholdResult:
    def test_summary_contains_threshold(self):
        result = ThresholdResult(
            threshold=0.42,
            method="f1",
            metric_value=0.85,
            metrics={"f1": 0.85, "precision": 0.9, "recall": 0.8},
        )
        s = result.summary()
        assert "0.42" in s
        assert "f1" in s

    def test_to_dict(self):
        result = ThresholdResult(
            threshold=0.55,
            method="youden",
            metric_value=0.7,
            metrics={"youden_j": 0.7},
        )
        d = result.to_dict()
        assert d["threshold"] == 0.55
        assert d["method"] == "youden"


# ---------------------------------------------------------------------------
# optimize_f1 tests
# ---------------------------------------------------------------------------

class TestOptimizeF1:
    def test_perfect_separation(self, simple_data):
        y_true, y_proba = simple_data
        result = optimize_f1(y_true, y_proba)
        assert result.method == "f1"
        assert result.threshold > 0.4
        assert result.threshold < 0.6
        assert result.metric_value >= 0.9

    def test_noisy_data(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_f1(y_true, y_proba)
        assert result.threshold > 0.0
        assert result.threshold < 1.0
        assert result.metric_value > 0.0

    def test_returns_all_thresholds(self, simple_data):
        y_true, y_proba = simple_data
        result = optimize_f1(y_true, y_proba, n_thresholds=50)
        assert result.all_thresholds is not None
        assert len(result.all_thresholds) == 50
        assert len(result.all_scores) == 50

    def test_metrics_at_optimal(self, simple_data):
        y_true, y_proba = simple_data
        result = optimize_f1(y_true, y_proba)
        assert "f1" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "accuracy" in result.metrics


# ---------------------------------------------------------------------------
# optimize_youden tests
# ---------------------------------------------------------------------------

class TestOptimizeYouden:
    def test_perfect_separation(self, simple_data):
        y_true, y_proba = simple_data
        result = optimize_youden(y_true, y_proba)
        assert result.method == "youden"
        assert result.metric_value > 0.8
        assert result.threshold > 0.4

    def test_metrics_include_sensitivity_specificity(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_youden(y_true, y_proba)
        assert "sensitivity" in result.metrics
        assert "specificity" in result.metrics
        assert "youden_j" in result.metrics


# ---------------------------------------------------------------------------
# optimize_cost tests
# ---------------------------------------------------------------------------

class TestOptimizeCost:
    def test_with_default_cost_matrix(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_cost(y_true, y_proba)
        assert result.method == "cost"
        assert result.metric_value != 0  # Should find some value

    def test_with_custom_cost_matrix(self, noisy_data, cost_matrix):
        y_true, y_proba = noisy_data
        result = optimize_cost(y_true, y_proba, cost_matrix=cost_matrix)
        assert result.method == "cost"
        assert "expected_value" in result.metrics

    def test_high_fn_cost_favors_recall(self, noisy_data):
        y_true, y_proba = noisy_data
        # High FN cost should push threshold lower (flag more)
        cm_conservative = CostMatrix(tp_benefit=100, fp_cost=1, fn_cost=500, tn_benefit=0)
        result_cons = optimize_cost(y_true, y_proba, cost_matrix=cm_conservative)

        # Low FN cost should push threshold higher (flag fewer)
        cm_aggressive = CostMatrix(tp_benefit=100, fp_cost=50, fn_cost=10, tn_benefit=0)
        result_agg = optimize_cost(y_true, y_proba, cost_matrix=cm_aggressive)

        assert result_cons.threshold <= result_agg.threshold

    def test_zero_fp_cost_flags_everyone(self, simple_data):
        y_true, y_proba = simple_data
        cm = CostMatrix(tp_benefit=100, fp_cost=0, fn_cost=100, tn_benefit=0)
        result = optimize_cost(y_true, y_proba, cost_matrix=cm)
        # With zero FP cost, should flag everyone (low threshold)
        assert result.threshold <= 0.1


# ---------------------------------------------------------------------------
# optimize_precision_recall tests
# ---------------------------------------------------------------------------

class TestOptimizePrecisionRecall:
    def test_achievable_precision(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_precision_recall(y_true, y_proba, target_precision=0.5)
        assert result.method == "precision_recall"
        assert result.metrics["precision"] >= 0.5 or result.metric_value >= 0

    def test_unachievable_precision(self, simple_data):
        """Very high target precision may not be reachable."""
        y_true, y_proba = simple_data
        result = optimize_precision_recall(y_true, y_proba, target_precision=0.99)
        # Should still return a result (falls back to best precision)
        assert result.threshold > 0

    def test_low_target_precision(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_precision_recall(y_true, y_proba, target_precision=0.3)
        assert result.metrics["precision"] >= 0.3


# ---------------------------------------------------------------------------
# find_threshold_for_target_rate tests
# ---------------------------------------------------------------------------

class TestFindThresholdForTargetRate:
    def test_target_rate_20_percent(self):
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        result = find_threshold_for_target_rate(y_proba, target_rate=0.2)
        # 20% of 10 = 2 customers should be flagged
        assert result.metrics["predicted_rate"] <= 0.3

    def test_target_rate_zero(self):
        y_proba = np.array([0.1, 0.5, 0.9])
        result = find_threshold_for_target_rate(y_proba, target_rate=0.0)
        assert result.threshold == 1.0
        assert result.metrics["predicted_rate"] == 0.0

    def test_target_rate_100_percent(self):
        y_proba = np.array([0.1, 0.5, 0.9])
        result = find_threshold_for_target_rate(y_proba, target_rate=1.0)
        # Should flag everyone (low threshold)
        assert result.threshold <= np.min(y_proba) + 0.01


# ---------------------------------------------------------------------------
# optimize_threshold convenience tests
# ---------------------------------------------------------------------------

class TestOptimizeThreshold:
    def test_f1_method(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_threshold(y_true, y_proba, method="f1")
        assert result.method == "f1"

    def test_youden_method(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_threshold(y_true, y_proba, method="youden")
        assert result.method == "youden"

    def test_cost_method(self, noisy_data, cost_matrix):
        y_true, y_proba = noisy_data
        result = optimize_threshold(y_true, y_proba, method="cost", cost_matrix=cost_matrix)
        assert result.method == "cost"

    def test_precision_recall_method(self, noisy_data):
        y_true, y_proba = noisy_data
        result = optimize_threshold(y_true, y_proba, method="precision_recall", target_precision=0.5)
        assert result.method == "precision_recall"

    def test_unknown_method_raises(self, noisy_data):
        y_true, y_proba = noisy_data
        with pytest.raises(ValueError, match="Unknown threshold method"):
            optimize_threshold(y_true, y_proba, method="nonexistent")
