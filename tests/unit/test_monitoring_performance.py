"""Tests for the monitoring performance module."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard.monitoring.performance import (
    MetricHistory,
    PerformanceAlert,
    PerformanceMonitor,
    PerformanceSnapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])


@pytest.fixture
def y_pred_good() -> np.ndarray:
    return np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])


@pytest.fixture
def y_pred_bad() -> np.ndarray:
    return np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])


@pytest.fixture
def y_proba() -> np.ndarray:
    return np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.6, 0.15])


# ---------------------------------------------------------------------------
# PerformanceSnapshot tests
# ---------------------------------------------------------------------------


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot."""

    def test_to_dict(self):
        snap = PerformanceSnapshot(
            timestamp="2025-01-01",
            model_name="test_model",
            n_samples=100,
            f1=0.85,
        )
        d = snap.to_dict()
        assert d["timestamp"] == "2025-01-01"
        assert d["f1"] == 0.85
        assert d["n_samples"] == 100

    def test_summary(self):
        snap = PerformanceSnapshot(
            timestamp="2025-01-01",
            model_name="test_model",
            f1=0.90,
            roc_auc=0.95,
        )
        s = snap.summary()
        assert "test_model" in s
        assert "F1" in s


# ---------------------------------------------------------------------------
# PerformanceAlert tests
# ---------------------------------------------------------------------------


class TestPerformanceAlert:
    """Tests for PerformanceAlert."""

    def test_summary(self):
        alert = PerformanceAlert(
            timestamp="2025-01-01",
            metric_name="f1",
            current_value=0.40,
            baseline_value=0.85,
            degradation_pct=52.94,
            severity="critical",
            model_name="test_model",
        )
        s = alert.summary()
        assert "CRITICAL" in s
        assert "f1" in s

    def test_to_dict(self):
        alert = PerformanceAlert(
            timestamp="2025-01-01",
            metric_name="f1",
            current_value=0.40,
            baseline_value=0.85,
            degradation_pct=52.94,
            severity="warning",
            model_name="test",
        )
        d = alert.to_dict()
        assert d["metric_name"] == "f1"
        assert d["severity"] == "warning"


# ---------------------------------------------------------------------------
# MetricHistory tests
# ---------------------------------------------------------------------------


class TestMetricHistory:
    """Tests for MetricHistory."""

    def test_add_and_latest(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.85)
        assert hist.latest() == 0.85

    def test_baseline(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.75)
        assert hist.baseline() == 0.80

    def test_mean(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.90)
        assert abs(hist.mean() - 0.85) < 1e-10

    def test_std(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.90)
        assert hist.std() > 0

    def test_trend_improving(self):
        hist = MetricHistory("f1")
        for i in range(10):
            hist.add(f"t{i}", 0.5 + i * 0.05)
        assert hist.trend() == "improving"

    def test_trend_degrading(self):
        hist = MetricHistory("f1")
        for i in range(10):
            hist.add(f"t{i}", 0.9 - i * 0.05)
        assert hist.trend() == "degrading"

    def test_trend_stable(self):
        hist = MetricHistory("f1")
        for i in range(10):
            hist.add(f"t{i}", 0.80)
        assert hist.trend() == "stable"

    def test_trend_too_few_points(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        assert hist.trend() == "stable"

    def test_len(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.85)
        assert len(hist) == 2

    def test_to_dataframe(self):
        hist = MetricHistory("f1")
        hist.add("t1", 0.80)
        hist.add("t2", 0.85)
        df = hist.to_dataframe()
        assert len(df) == 2
        assert "f1" in df.columns

    def test_empty(self):
        hist = MetricHistory("f1")
        assert hist.latest() is None
        assert hist.baseline() is None
        assert hist.mean() == 0.0
        assert hist.std() == 0.0
        assert len(hist) == 0


# ---------------------------------------------------------------------------
# PerformanceMonitor tests
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_evaluate_basic(self, y_true, y_pred_good, y_proba):
        """Basic evaluation should produce a snapshot."""
        monitor = PerformanceMonitor(model_name="test")
        snap = monitor.evaluate(y_true, y_pred_good, y_proba)
        assert snap.model_name == "test"
        assert snap.f1 > 0
        assert snap.roc_auc > 0
        assert snap.n_samples == 10

    def test_evaluate_without_proba(self, y_true, y_pred_good):
        """Evaluation without probabilities should set AUC to 0."""
        monitor = PerformanceMonitor(model_name="test")
        snap = monitor.evaluate(y_true, y_pred_good, y_proba=None)
        assert snap.roc_auc == 0.0
        assert snap.pr_auc == 0.0

    def test_custom_timestamp(self, y_true, y_pred_good):
        """Custom timestamp should be used."""
        monitor = PerformanceMonitor()
        snap = monitor.evaluate(y_true, y_pred_good, timestamp="2025-01-01")
        assert snap.timestamp == "2025-01-01"

    def test_snapshot_history(self, y_true, y_pred_good):
        """Multiple evaluations should build history."""
        monitor = PerformanceMonitor()
        monitor.evaluate(y_true, y_pred_good, timestamp="t1")
        monitor.evaluate(y_true, y_pred_good, timestamp="t2")
        assert len(monitor.snapshots) == 2

    def test_degradation_alert(self, y_true, y_pred_good, y_pred_bad):
        """Degraded predictions should trigger alerts."""
        monitor = PerformanceMonitor(
            model_name="test",
            warning_threshold_pct=5.0,
            critical_threshold_pct=15.0,
        )
        # Baseline: good predictions
        monitor.evaluate(y_true, y_pred_good, timestamp="baseline")

        # Current: bad predictions (much worse)
        y_true_large = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5)
        y_pred_bad_large = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 5)
        monitor.evaluate(y_true_large, y_pred_bad_large, timestamp="current")

        # Should have alerts
        assert len(monitor.alerts) > 0

    def test_get_trends(self, y_true, y_pred_good):
        """Trends should be available after multiple evaluations."""
        monitor = PerformanceMonitor()
        for i in range(5):
            monitor.evaluate(y_true, y_pred_good, timestamp=f"t{i}")

        trends = monitor.get_trends()
        assert "f1" in trends
        assert trends["f1"] == "stable"  # Same predictions each time

    def test_summary(self, y_true, y_pred_good):
        """Summary should be a non-empty string."""
        monitor = PerformanceMonitor()
        monitor.evaluate(y_true, y_pred_good)
        s = monitor.summary()
        assert len(s) > 0

    def test_to_dataframe(self, y_true, y_pred_good):
        """Should produce a valid DataFrame."""
        monitor = PerformanceMonitor()
        monitor.evaluate(y_true, y_pred_good, timestamp="t1")
        df = monitor.to_dataframe()
        assert len(df) == 1
        assert "f1" in df.columns

    def test_alerts_to_dataframe(self, y_true, y_pred_good, y_pred_bad):
        """Alerts DataFrame should be produced."""
        monitor = PerformanceMonitor()
        monitor.evaluate(y_true, y_pred_good, timestamp="baseline")

        y_true_large = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5)
        y_pred_bad_large = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 5)
        monitor.evaluate(y_true_large, y_pred_bad_large, timestamp="current")

        df = monitor.alerts_to_dataframe()
        assert len(df) > 0
        assert "metric_name" in df.columns

    def test_custom_metrics(self, y_true, y_pred_good):
        """Custom metrics_to_track should be respected."""
        monitor = PerformanceMonitor(metrics_to_track=["f1", "roc_auc"])
        monitor.evaluate(y_true, y_pred_good)
        assert "f1" in monitor.history
        assert "roc_auc" in monitor.history
        assert "accuracy" not in monitor.history

    def test_metadata(self, y_true, y_pred_good):
        """Metadata should be stored in the snapshot."""
        monitor = PerformanceMonitor()
        snap = monitor.evaluate(
            y_true,
            y_pred_good,
            metadata={"data_source": "production", "version": "2.0"},
        )
        assert snap.metadata["data_source"] == "production"
        assert snap.metadata["version"] == "2.0"

    def test_churn_rate(self, y_true, y_pred_good):
        """Churn rate should be computed from y_true."""
        monitor = PerformanceMonitor()
        snap = monitor.evaluate(y_true, y_pred_good)
        expected_rate = y_true.mean()
        assert abs(snap.churn_rate - expected_rate) < 1e-10

    def test_no_alerts_first_snapshot(self, y_true, y_pred_good):
        """First snapshot should never trigger degradation alerts."""
        monitor = PerformanceMonitor()
        monitor.evaluate(y_true, y_pred_good)
        assert len(monitor.alerts) == 0
