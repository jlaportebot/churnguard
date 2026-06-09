"""ChurnGuard Monitoring — drift detection, performance tracking, and alerting.

This package provides tools for monitoring deployed churn prediction models:

- :mod:`~churnguard.monitoring.drift` — Data drift detection via PSI, KS, and
  chi-square tests.
- :mod:`~churnguard.monitoring.concept` — Concept drift detection via ADWIN,
  DDM, and EDDM streaming algorithms.
- :mod:`~churnguard.monitoring.performance` — Model performance monitoring with
  historical trend tracking and degradation alerts.
- :mod:`~churnguard.monitoring.alerts` — Configurable alert rules, actions, and
  alert history management.
- :mod:`~churnguard.monitoring.report` — Self-contained HTML dashboard
  generation for monitoring results.

Quick start
-----------

>>> from churnguard.monitoring import DataDriftDetector, PerformanceMonitor, AlertManager
>>> detector = DataDriftDetector()
>>> result = detector.detect(reference_df, current_df)
>>> print(result.summary())
"""

from __future__ import annotations

from churnguard.monitoring.alerts import (
    DEFAULT_CHURN_ALERT_RULES,
    Alert,
    AlertAction,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackAlertAction,
    FileAlertAction,
    LogAlertAction,
)
from churnguard.monitoring.concept import (
    ADWIN,
    DDM,
    EDDM,
    ConceptDriftDetector,
    ConceptDriftResult,
    DriftState,
)
from churnguard.monitoring.drift import (
    ChiSquareResult,
    DataDriftDetector,
    DriftResult,
    DriftSeverity,
    KSTestResult,
    PSIResult,
    compute_psi,
)
from churnguard.monitoring.performance import (
    MetricHistory,
    PerformanceAlert,
    PerformanceMonitor,
    PerformanceSnapshot,
)
from churnguard.monitoring.report import (
    MonitoringReport,
    MonitoringReportConfig,
)

__all__ = [
    # Drift
    "DataDriftDetector",
    "DriftResult",
    "DriftSeverity",
    "PSIResult",
    "KSTestResult",
    "ChiSquareResult",
    "compute_psi",
    # Concept drift
    "ADWIN",
    "DDM",
    "EDDM",
    "ConceptDriftDetector",
    "ConceptDriftResult",
    "DriftState",
    # Performance
    "PerformanceMonitor",
    "PerformanceSnapshot",
    "PerformanceAlert",
    "MetricHistory",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertAction",
    "LogAlertAction",
    "FileAlertAction",
    "CallbackAlertAction",
    "DEFAULT_CHURN_ALERT_RULES",
    # Report
    "MonitoringReport",
    "MonitoringReportConfig",
]
