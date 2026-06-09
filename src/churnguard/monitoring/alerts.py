"""Alert management for model monitoring.

Provides configurable alert rules, multi-level severity, and
action hooks for responding to drift and performance degradation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Alert data
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """An alert triggered by monitoring.

    Attributes:
        alert_id: Unique identifier for the alert.
        timestamp: When the alert was generated (ISO format).
        rule_name: Name of the rule that triggered the alert.
        severity: Alert severity level.
        metric_name: Name of the metric that triggered the alert.
        current_value: Current metric value.
        threshold: Threshold that was crossed.
        message: Human-readable alert message.
        details: Additional details about the alert.
        acknowledged: Whether the alert has been acknowledged.
    """

    alert_id: str
    timestamp: str
    rule_name: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.severity.value.upper()}] {self.rule_name}: "
            f"{self.metric_name}={self.current_value:.4f} "
            f"(threshold={self.threshold:.4f}) — {self.message}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 6),
            "threshold": round(self.threshold, 6),
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }

    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.acknowledged = True


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """Configuration for an alert rule.

    Attributes:
        name: Human-readable name for the rule.
        metric_name: Name of the metric to monitor.
        condition: Comparison operator: 'gt', 'lt', 'gte', 'lte', 'eq'.
        threshold: Threshold value.
        severity: Severity level when triggered.
        message: Alert message template. Use {metric}, {value}, {threshold}.
        cooldown_minutes: Minimum minutes between repeated alerts.
        enabled: Whether the rule is active.
    """

    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = "{metric} is {condition} {threshold} (current: {value})"
    cooldown_minutes: int = 60
    enabled: bool = True

    def check(self, value: float) -> bool:
        """Check if the value triggers this rule.

        Parameters
        ----------
        value : float
            Current metric value.

        Returns
        -------
        bool
            True if the rule is triggered.
        """
        if not self.enabled:
            return False

        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        op = ops.get(self.condition)
        if op is None:
            logger.warning("Unknown condition '%s' for rule '%s'", self.condition, self.name)
            return False
        return op(value, self.threshold)

    def format_message(self, value: float) -> str:
        """Format the alert message with current values."""
        return self.message.format(
            metric=self.metric_name,
            condition=self.condition,
            threshold=self.threshold,
            value=value,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "cooldown_minutes": self.cooldown_minutes,
            "enabled": self.enabled,
        }


# ---------------------------------------------------------------------------
# Alert actions
# ---------------------------------------------------------------------------


class AlertAction(ABC):
    """Abstract base class for alert actions.

    Subclass this to define custom actions that run when an alert fires.
    """

    @abstractmethod
    def execute(self, alert: Alert) -> None:
        """Execute the alert action.

        Parameters
        ----------
        alert : Alert
            The alert that triggered this action.
        """


class LogAlertAction(AlertAction):
    """Log the alert to the Python logger."""

    def __init__(self, log_level: str = "WARNING") -> None:
        self.log_level = log_level

    def execute(self, alert: Alert) -> None:
        level = getattr(logging, self.log_level, logging.WARNING)
        logger.log(level, "ALERT: %s", alert.summary())


class FileAlertAction(AlertAction):
    """Append the alert to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, alert: Alert) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(alert.to_dict(), default=str) + "\n")


class CallbackAlertAction(AlertAction):
    """Call a custom function when an alert fires."""

    def __init__(self, callback: Callable[[Alert], None]) -> None:
        self.callback = callback

    def execute(self, alert: Alert) -> None:
        try:
            self.callback(alert)
        except Exception as e:
            logger.error("Alert callback failed: %s", e)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Manage alert rules, dispatch actions, and track alert history.

    Parameters
    ----------
    rules : list of AlertRule, optional
        Initial alert rules.
    actions : list of AlertAction, optional
        Global actions that fire for every alert.
    max_history : int
        Maximum number of alerts to keep in history (default 1000).
    """

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        actions: list[AlertAction] | None = None,
        max_history: int = 1000,
    ) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._actions: list[AlertAction] = actions or [LogAlertAction()]
        self._rule_actions: dict[str, list[AlertAction]] = {}
        self._history: list[Alert] = []
        self._last_fired: dict[str, str] = {}  # rule_name → last timestamp
        self._alert_counter = 0
        self.max_history = max_history

        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Parameters
        ----------
        rule : AlertRule
            The rule to add.
        """
        self._rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule by name.

        Parameters
        ----------
        rule_name : str
            Name of the rule to remove.
        """
        self._rules.pop(rule_name, None)
        self._rule_actions.pop(rule_name, None)

    def add_action(self, action: AlertAction, rule_name: str | None = None) -> None:
        """Add an alert action.

        Parameters
        ----------
        action : AlertAction
            The action to add.
        rule_name : str, optional
            If provided, action only fires for this rule.
            Otherwise, it fires for all alerts.
        """
        if rule_name:
            if rule_name not in self._rule_actions:
                self._rule_actions[rule_name] = []
            self._rule_actions[rule_name].append(action)
        else:
            self._actions.append(action)

    def check(
        self,
        metrics: dict[str, float],
        timestamp: str | None = None,
    ) -> list[Alert]:
        """Check all rules against current metric values.

        Parameters
        ----------
        metrics : dict
            Metric name → current value mapping.
        timestamp : str, optional
            ISO timestamp. Defaults to now.

        Returns
        -------
        list of Alert
            New alerts that were triggered.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        new_alerts: list[Alert] = []

        for rule_name, rule in self._rules.items():
            if not rule.enabled:
                continue

            value = metrics.get(rule.metric_name)
            if value is None:
                continue

            if rule.check(value):
                # Check cooldown
                last_fired = self._last_fired.get(rule_name)
                if last_fired and rule.cooldown_minutes > 0:
                    last_dt = datetime.fromisoformat(last_fired)
                    now_dt = datetime.fromisoformat(timestamp)
                    delta_minutes = (now_dt - last_dt).total_seconds() / 60
                    if delta_minutes < rule.cooldown_minutes:
                        continue

                # Create alert
                self._alert_counter += 1
                alert = Alert(
                    alert_id=f"alert-{self._alert_counter:06d}",
                    timestamp=timestamp,
                    rule_name=rule_name,
                    severity=rule.severity,
                    metric_name=rule.metric_name,
                    current_value=value,
                    threshold=rule.threshold,
                    message=rule.format_message(value),
                )

                new_alerts.append(alert)
                self._history.append(alert)
                self._last_fired[rule_name] = timestamp

                # Execute actions
                self._execute_actions(alert, rule_name)

        # Trim history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return new_alerts

    def _execute_actions(self, alert: Alert, rule_name: str) -> None:
        """Execute all actions for an alert."""
        # Global actions
        for action in self._actions:
            try:
                action.execute(alert)
            except Exception as e:
                logger.error("Alert action failed: %s", e)

        # Rule-specific actions
        for action in self._rule_actions.get(rule_name, []):
            try:
                action.execute(alert)
            except Exception as e:
                logger.error("Rule action failed: %s", e)

    @property
    def history(self) -> list[Alert]:
        """All alerts in history."""
        return self._history

    @property
    def unacknowledged(self) -> list[Alert]:
        """All unacknowledged alerts."""
        return [a for a in self._history if not a.acknowledged]

    @property
    def rules(self) -> dict[str, AlertRule]:
        """Current alert rules."""
        return self._rules

    def acknowledge_all(self) -> int:
        """Acknowledge all unacknowledged alerts.

        Returns
        -------
        int
            Number of alerts acknowledged.
        """
        count = 0
        for alert in self._history:
            if not alert.acknowledged:
                alert.acknowledge()
                count += 1
        return count

    def summary(self) -> str:
        """Human-readable summary."""
        n_total = len(self._history)
        n_unack = len(self.unacknowledged)
        n_critical = sum(1 for a in self._history if a.severity == AlertSeverity.CRITICAL)

        lines = [
            "=== Alert Manager ===",
            f"Rules: {len(self._rules)}",
            f"Total alerts: {n_total}",
            f"Unacknowledged: {n_unack}",
            f"Critical: {n_critical}",
        ]

        if self.unacknowledged:
            lines.append("")
            lines.append("Unacknowledged alerts:")
            for alert in self.unacknowledged[-10:]:
                lines.append(f"  {alert.summary()}")

        return "\n".join(lines)

    def to_dataframe(self):
        """Convert alert history to a pandas DataFrame."""
        import pandas as pd

        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._history])


# ---------------------------------------------------------------------------
# Default alert rules for churn monitoring
# ---------------------------------------------------------------------------

DEFAULT_CHURN_ALERT_RULES: list[AlertRule] = [
    AlertRule(
        name="f1_degradation_warning",
        metric_name="f1",
        condition="lt",
        threshold=0.60,
        severity=AlertSeverity.WARNING,
        message="F1 score has dropped below 0.60 (current: {value:.4f})",
        cooldown_minutes=120,
    ),
    AlertRule(
        name="f1_degradation_critical",
        metric_name="f1",
        condition="lt",
        threshold=0.40,
        severity=AlertSeverity.CRITICAL,
        message="F1 score critically low (current: {value:.4f})",
        cooldown_minutes=30,
    ),
    AlertRule(
        name="roc_auc_degradation",
        metric_name="roc_auc",
        condition="lt",
        threshold=0.70,
        severity=AlertSeverity.WARNING,
        message="ROC AUC below 0.70 (current: {value:.4f})",
        cooldown_minutes=120,
    ),
    AlertRule(
        name="drift_score_high",
        metric_name="drift_score",
        condition="gte",
        threshold=0.25,
        severity=AlertSeverity.WARNING,
        message="Data drift score high (current: {value:.4f})",
        cooldown_minutes=1440,
    ),
    AlertRule(
        name="drift_score_critical",
        metric_name="drift_score",
        condition="gte",
        threshold=0.50,
        severity=AlertSeverity.CRITICAL,
        message="Data drift score critical (current: {value:.4f})",
        cooldown_minutes=60,
    ),
    AlertRule(
        name="churn_rate_shift",
        metric_name="churn_rate",
        condition="gt",
        threshold=0.40,
        severity=AlertSeverity.WARNING,
        message="Churn rate unusually high (current: {value:.1%})",
        cooldown_minutes=1440,
    ),
]
