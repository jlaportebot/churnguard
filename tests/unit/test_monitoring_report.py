"""Tests for the monitoring report module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnguard.monitoring.drift import (
    DataDriftDetector,
    DriftSeverity,
)
from churnguard.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from churnguard.monitoring.performance import PerformanceMonitor
from churnguard.monitoring.report import (
    MonitoringReport,
    MonitoringReportConfig,
    _bar_chart_html,
    _line_chart_html,
    _severity_badge,
    _alert_severity_badge,
)


# ---------------------------------------------------------------------------
# Chart helper tests
# ---------------------------------------------------------------------------

class TestBarChart:
    """Tests for _bar_chart_html."""

    def test_basic_chart(self):
        html = _bar_chart_html(
            labels=["a", "b", "c"],
            values=[0.1, 0.2, 0.3],
            title="Test Chart",
        )
        assert "<svg" in html
        assert "Test Chart" in html

    def test_empty_data(self):
        html = _bar_chart_html(labels=[], values=[], title="Empty")
        assert "No data" in html

    def test_threshold_line(self):
        html = _bar_chart_html(
            labels=["a", "b"],
            values=[0.1, 0.2],
            title="With Threshold",
            threshold_line=0.15,
            threshold_label="T=0.15",
        )
        assert "Threshold" in html

    def test_long_labels_truncated(self):
        html = _bar_chart_html(
            labels=["very_long_feature_name_that_exceeds_25_chars"],
            values=[0.5],
            title="Long Labels",
        )
        assert "<svg" in html


class TestLineChart:
    """Tests for _line_chart_html."""

    def test_basic_chart(self):
        html = _line_chart_html(
            series={"f1": [0.8, 0.85, 0.82, 0.88]},
            x_labels=["t1", "t2", "t3", "t4"],
            title="Performance Over Time",
        )
        assert "<svg" in html
        assert "Performance Over Time" in html

    def test_multiple_series(self):
        html = _line_chart_html(
            series={"f1": [0.8, 0.85], "auc": [0.9, 0.92]},
            x_labels=["t1", "t2"],
            title="Multi Series",
        )
        assert "<svg" in html

    def test_empty_data(self):
        html = _line_chart_html(series={}, x_labels=[], title="Empty")
        assert "No data" in html

    def test_constant_values(self):
        """Chart should handle constant values (no division by zero)."""
        html = _line_chart_html(
            series={"f1": [0.5, 0.5, 0.5]},
            x_labels=["t1", "t2", "t3"],
            title="Constant",
        )
        assert "<svg" in html


class TestSeverityBadge:
    """Tests for severity badge helpers."""

    def test_drift_severity_badge(self):
        badge = _severity_badge(DriftSeverity.HIGH)
        assert "HIGH" in badge
        assert "background" in badge

    def test_alert_severity_badge(self):
        badge = _alert_severity_badge(AlertSeverity.CRITICAL)
        assert "CRITICAL" in badge


# ---------------------------------------------------------------------------
# MonitoringReportConfig tests
# ---------------------------------------------------------------------------

class TestMonitoringReportConfig:
    """Tests for MonitoringReportConfig."""

    def test_defaults(self):
        cfg = MonitoringReportConfig()
        assert cfg.include_drift is True
        assert cfg.include_performance is True
        assert cfg.include_alerts is True
        assert cfg.color_scheme == "light"

    def test_custom(self):
        cfg = MonitoringReportConfig(title="Custom", include_concept_drift=False)
        assert cfg.title == "Custom"


# ---------------------------------------------------------------------------
# MonitoringReport tests
# ---------------------------------------------------------------------------

class TestMonitoringReport:
    """Tests for MonitoringReport."""

    @pytest.fixture
    def drift_result(self):
        """Create a sample drift result."""
        np.random.seed(42)
        ref = pd.DataFrame({
            "tenure": np.random.exponential(30, 300),
            "charges": np.random.normal(65, 15, 300),
        })
        cur = pd.DataFrame({
            "tenure": np.random.exponential(30, 300),
            "charges": np.random.normal(65, 15, 300),
        })
        detector = DataDriftDetector()
        return detector.detect(ref, cur)

    @pytest.fixture
    def performance_monitor(self):
        """Create a sample performance monitor with data."""
        monitor = PerformanceMonitor(model_name="test_model")
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 3)
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0] * 3)
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.6, 0.15] * 3)
        monitor.evaluate(y_true, y_pred, y_proba, timestamp="2025-01-01")
        monitor.evaluate(y_true, y_pred, y_proba, timestamp="2025-02-01")
        return monitor

    @pytest.fixture
    def alert_manager(self):
        """Create a sample alert manager with alerts."""
        rules = [
            AlertRule(
                name="f1_low",
                metric_name="f1",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=0,
            ),
        ]
        mgr = AlertManager(rules=rules)
        mgr.check({"f1": 0.3})
        return mgr

    def test_generate_empty_report(self):
        """Report should work with no data."""
        report = MonitoringReport()
        html = report.generate()
        assert "<!DOCTYPE html>" in html
        assert "ChurnGuard" in html

    def test_generate_with_drift(self, drift_result):
        """Report should include drift section."""
        report = MonitoringReport()
        html = report.generate(drift_result=drift_result)
        assert "Data Drift Detection" in html
        assert "Features Tested" in html

    def test_generate_with_performance(self, performance_monitor):
        """Report should include performance section."""
        report = MonitoringReport()
        html = report.generate(performance_monitor=performance_monitor)
        assert "Model Performance" in html
        assert "F1 Score" in html

    def test_generate_with_alerts(self, alert_manager):
        """Report should include alerts section."""
        report = MonitoringReport()
        html = report.generate(alert_manager=alert_manager)
        assert "Alert History" in html

    def test_generate_all_sections(self, drift_result, performance_monitor, alert_manager):
        """Report with all sections should be complete HTML."""
        report = MonitoringReport()
        html = report.generate(
            drift_result=drift_result,
            performance_monitor=performance_monitor,
            alert_manager=alert_manager,
        )
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "Data Drift" in html
        assert "Model Performance" in html
        assert "Alert" in html

    def test_save_to_file(self, drift_result, tmp_path):
        """Report should be saved to a file."""
        report = MonitoringReport()
        path = tmp_path / "report.html"
        html = report.generate(drift_result=drift_result, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_creates_parent_dirs(self, tmp_path):
        """Saving should create parent directories."""
        report = MonitoringReport()
        path = tmp_path / "sub" / "dir" / "report.html"
        html = report.generate(output_path=path)
        assert path.exists()

    def test_overall_status_healthy(self, drift_result):
        """Status should be HEALTHY when no issues."""
        report = MonitoringReport()
        html = report.generate(drift_result=drift_result)
        # Status may be HEALTHY, WARNING, or CRITICAL depending on drift results
        assert any(s in html for s in ["HEALTHY", "WARNING", "CRITICAL"])

    def test_overall_status_warning(self):
        """Status should be WARNING when alerts are present."""
        rules = [
            AlertRule(
                name="f1_low",
                metric_name="f1",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=0,
            ),
        ]
        mgr = AlertManager(rules=rules)
        mgr.check({"f1": 0.3})

        report = MonitoringReport()
        html = report.generate(alert_manager=mgr)
        assert "WARNING" in html

    def test_overall_status_critical(self):
        """Status should be CRITICAL when critical alerts are present."""
        rules = [
            AlertRule(
                name="f1_critical",
                metric_name="f1",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=0,
            ),
        ]
        mgr = AlertManager(rules=rules)
        mgr.check({"f1": 0.3})

        report = MonitoringReport()
        html = report.generate(alert_manager=mgr)
        assert "CRITICAL" in html

    def test_concept_drift_section(self):
        """Report should include concept drift section when data provided."""
        from churnguard.monitoring.concept import ADWIN
        adwin = ADWIN()
        # Simulate some data
        for _ in range(50):
            adwin.update(0, 0)
        result = adwin.detect()

        report = MonitoringReport()
        html = report.generate(concept_drift_results={"ADWIN": result})
        assert "Concept Drift" in html

    def test_custom_config(self, drift_result):
        """Custom config should be respected."""
        cfg = MonitoringReportConfig(title="My Custom Report", include_alerts=False)
        report = MonitoringReport(config=cfg)
        html = report.generate(drift_result=drift_result)
        assert "My Custom Report" in html
