"""Tests for the monitoring alerts module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from churnguard.monitoring.alerts import (
    Alert,
    AlertAction,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackAlertAction,
    FileAlertAction,
    LogAlertAction,
    DEFAULT_CHURN_ALERT_RULES,
)


# ---------------------------------------------------------------------------
# AlertSeverity tests
# ---------------------------------------------------------------------------

class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_str(self):
        assert str(AlertSeverity.WARNING) == "warning"


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------

class TestAlert:
    """Tests for Alert data class."""

    def test_to_dict(self):
        alert = Alert(
            alert_id="alert-001",
            timestamp="2025-01-01",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            metric_name="f1",
            current_value=0.45,
            threshold=0.60,
            message="F1 below threshold",
        )
        d = alert.to_dict()
        assert d["alert_id"] == "alert-001"
        assert d["severity"] == "warning"

    def test_summary(self):
        alert = Alert(
            alert_id="alert-001",
            timestamp="2025-01-01",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            metric_name="drift_score",
            current_value=0.60,
            threshold=0.50,
            message="Drift score too high",
        )
        s = alert.summary()
        assert "CRITICAL" in s
        assert "drift_score" in s

    def test_acknowledge(self):
        alert = Alert(
            alert_id="alert-001",
            timestamp="2025-01-01",
            rule_name="r",
            severity=AlertSeverity.WARNING,
            metric_name="f1",
            current_value=0.3,
            threshold=0.6,
            message="msg",
        )
        assert not alert.acknowledged
        alert.acknowledge()
        assert alert.acknowledged


# ---------------------------------------------------------------------------
# AlertRule tests
# ---------------------------------------------------------------------------

class TestAlertRule:
    """Tests for AlertRule data class."""

    def test_check_gt(self):
        rule = AlertRule(name="test", metric_name="f1", condition="gt", threshold=0.5)
        assert rule.check(0.6) is True
        assert rule.check(0.4) is False

    def test_check_lt(self):
        rule = AlertRule(name="test", metric_name="f1", condition="lt", threshold=0.5)
        assert rule.check(0.4) is True
        assert rule.check(0.6) is False

    def test_check_gte(self):
        rule = AlertRule(name="test", metric_name="f1", condition="gte", threshold=0.5)
        assert rule.check(0.5) is True
        assert rule.check(0.49) is False

    def test_check_lte(self):
        rule = AlertRule(name="test", metric_name="f1", condition="lte", threshold=0.5)
        assert rule.check(0.5) is True
        assert rule.check(0.51) is False

    def test_check_eq(self):
        rule = AlertRule(name="test", metric_name="f1", condition="eq", threshold=0.5)
        assert rule.check(0.5) is True
        assert rule.check(0.6) is False

    def test_check_disabled(self):
        rule = AlertRule(name="test", metric_name="f1", condition="lt", threshold=0.5, enabled=False)
        assert rule.check(0.3) is False

    def test_unknown_condition(self):
        rule = AlertRule(name="test", metric_name="f1", condition="unknown", threshold=0.5)
        assert rule.check(0.3) is False

    def test_format_message(self):
        rule = AlertRule(
            name="test", metric_name="f1", condition="lt",
            threshold=0.5,
            message="{metric} is {condition} {threshold} (current: {value})",
        )
        msg = rule.format_message(0.3)
        assert "f1" in msg
        assert "0.3" in msg

    def test_to_dict(self):
        rule = AlertRule(
            name="test", metric_name="f1", condition="lt",
            threshold=0.5, severity=AlertSeverity.CRITICAL,
        )
        d = rule.to_dict()
        assert d["name"] == "test"
        assert d["severity"] == "critical"


# ---------------------------------------------------------------------------
# AlertAction tests
# ---------------------------------------------------------------------------

class TestLogAlertAction:
    """Tests for LogAlertAction."""

    def test_execute(self):
        action = LogAlertAction(log_level="WARNING")
        alert = Alert(
            alert_id="a1", timestamp="t", rule_name="r",
            severity=AlertSeverity.WARNING, metric_name="f1",
            current_value=0.3, threshold=0.6, message="msg",
        )
        # Should not raise
        action.execute(alert)


class TestFileAlertAction:
    """Tests for FileAlertAction."""

    def test_execute(self, tmp_path):
        path = tmp_path / "alerts.jsonl"
        action = FileAlertAction(path)
        alert = Alert(
            alert_id="a1", timestamp="2025-01-01", rule_name="r",
            severity=AlertSeverity.WARNING, metric_name="f1",
            current_value=0.3, threshold=0.6, message="msg",
        )
        action.execute(alert)
        assert path.exists()
        content = path.read_text().strip()
        data = json.loads(content)
        assert data["alert_id"] == "a1"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "alerts.jsonl"
        action = FileAlertAction(path)
        alert = Alert(
            alert_id="a1", timestamp="t", rule_name="r",
            severity=AlertSeverity.INFO, metric_name="x",
            current_value=1.0, threshold=0.5, message="m",
        )
        action.execute(alert)
        assert path.exists()


class TestCallbackAlertAction:
    """Tests for CallbackAlertAction."""

    def test_execute(self):
        received = []
        action = CallbackAlertAction(callback=lambda a: received.append(a))
        alert = Alert(
            alert_id="a1", timestamp="t", rule_name="r",
            severity=AlertSeverity.WARNING, metric_name="f1",
            current_value=0.3, threshold=0.6, message="msg",
        )
        action.execute(alert)
        assert len(received) == 1

    def test_exception_handled(self):
        """Callback exceptions should be caught gracefully."""
        action = CallbackAlertAction(callback=lambda a: 1 / 0)
        alert = Alert(
            alert_id="a1", timestamp="t", rule_name="r",
            severity=AlertSeverity.WARNING, metric_name="f1",
            current_value=0.3, threshold=0.6, message="msg",
        )
        # Should not raise
        action.execute(alert)


class TestAlertActionABC:
    """Test that AlertAction cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AlertAction()


# ---------------------------------------------------------------------------
# AlertManager tests
# ---------------------------------------------------------------------------

class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def simple_rules(self):
        return [
            AlertRule(
                name="f1_low",
                metric_name="f1",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=0,
            ),
            AlertRule(
                name="drift_high",
                metric_name="drift_score",
                condition="gte",
                threshold=0.25,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=0,
            ),
        ]

    def test_add_rule(self, simple_rules):
        mgr = AlertManager()
        for rule in simple_rules:
            mgr.add_rule(rule)
        assert len(mgr.rules) == 2

    def test_remove_rule(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        mgr.remove_rule("f1_low")
        assert "f1_low" not in mgr.rules

    def test_check_triggers_alert(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        alerts = mgr.check({"f1": 0.3, "drift_score": 0.1})
        assert len(alerts) == 1
        assert alerts[0].rule_name == "f1_low"

    def test_check_multiple_alerts(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        alerts = mgr.check({"f1": 0.3, "drift_score": 0.5})
        assert len(alerts) == 2

    def test_check_no_alerts(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        alerts = mgr.check({"f1": 0.8, "drift_score": 0.05})
        assert len(alerts) == 0

    def test_check_missing_metric(self, simple_rules):
        """Missing metrics should be silently skipped."""
        mgr = AlertManager(rules=simple_rules)
        alerts = mgr.check({"f1": 0.8})  # drift_score missing
        assert len(alerts) == 0

    def test_cooldown(self):
        """Alerts should respect cooldown period."""
        rule = AlertRule(
            name="f1_low",
            metric_name="f1",
            condition="lt",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=60,
        )
        mgr = AlertManager(rules=[rule])
        alerts1 = mgr.check({"f1": 0.3}, timestamp="2025-01-01T10:00:00")
        assert len(alerts1) == 1

        # Same time window → should not fire again
        alerts2 = mgr.check({"f1": 0.3}, timestamp="2025-01-01T10:30:00")
        assert len(alerts2) == 0

        # After cooldown → should fire
        alerts3 = mgr.check({"f1": 0.3}, timestamp="2025-01-01T11:30:00")
        assert len(alerts3) == 1

    def test_zero_cooldown(self):
        """Zero cooldown should allow repeated alerts."""
        rule = AlertRule(
            name="f1_low",
            metric_name="f1",
            condition="lt",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=0,
        )
        mgr = AlertManager(rules=[rule])
        alerts1 = mgr.check({"f1": 0.3})
        alerts2 = mgr.check({"f1": 0.3})
        assert len(alerts1) == 1
        assert len(alerts2) == 1

    def test_unacknowledged(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        mgr.check({"f1": 0.3, "drift_score": 0.5})
        assert len(mgr.unacknowledged) == 2

    def test_acknowledge_all(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        mgr.check({"f1": 0.3, "drift_score": 0.5})
        count = mgr.acknowledge_all()
        assert count == 2
        assert len(mgr.unacknowledged) == 0

    def test_history(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        mgr.check({"f1": 0.3, "drift_score": 0.5})
        assert len(mgr.history) == 2

    def test_max_history(self):
        """History should be trimmed when it exceeds max."""
        rule = AlertRule(
            name="f1_low",
            metric_name="f1",
            condition="lt",
            threshold=0.5,
            cooldown_minutes=0,
        )
        mgr = AlertManager(rules=[rule], max_history=5)
        for i in range(10):
            mgr.check({"f1": 0.3})
        assert len(mgr.history) == 5

    def test_add_global_action(self, simple_rules):
        received = []
        mgr = AlertManager(rules=simple_rules)
        mgr.add_action(CallbackAlertAction(callback=lambda a: received.append(a)))
        mgr.check({"f1": 0.3})
        assert len(received) == 1

    def test_add_rule_specific_action(self, simple_rules):
        received = []
        mgr = AlertManager(rules=simple_rules)
        mgr.add_action(
            CallbackAlertAction(callback=lambda a: received.append("drift")),
            rule_name="drift_high",
        )
        mgr.check({"f1": 0.3, "drift_score": 0.5})
        assert len(received) == 1
        # The rule-specific action should only fire for drift_high

    def test_summary(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        mgr.check({"f1": 0.3})
        s = mgr.summary()
        assert "Alert Manager" in s

    def test_custom_timestamp(self, simple_rules):
        mgr = AlertManager(rules=simple_rules)
        alerts = mgr.check({"f1": 0.3}, timestamp="2025-06-01")
        assert alerts[0].timestamp == "2025-06-01"


# ---------------------------------------------------------------------------
# Default alert rules tests
# ---------------------------------------------------------------------------

class TestDefaultChurnAlertRules:
    """Tests for DEFAULT_CHURN_ALERT_RULES."""

    def test_rules_exist(self):
        assert len(DEFAULT_CHURN_ALERT_RULES) > 0

    def test_rule_names_unique(self):
        names = [r.name for r in DEFAULT_CHURN_ALERT_RULES]
        assert len(names) == len(set(names))

    def test_rules_are_enabled(self):
        for rule in DEFAULT_CHURN_ALERT_RULES:
            assert rule.enabled is True

    def test_rules_have_valid_conditions(self):
        valid_conditions = {"gt", "lt", "gte", "lte", "eq"}
        for rule in DEFAULT_CHURN_ALERT_RULES:
            assert rule.condition in valid_conditions
