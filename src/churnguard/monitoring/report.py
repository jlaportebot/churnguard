"""Monitoring report generation for ChurnGuard.

Generates standalone HTML dashboards that visualize drift detection
results, performance trends, and alert history. Reports are
self-contained (no external dependencies) and can be shared via
email or file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from churnguard.monitoring.alerts import AlertManager, AlertSeverity
from churnguard.monitoring.drift import (
    DriftResult,
    DriftSeverity,
)
from churnguard.monitoring.performance import (
    PerformanceMonitor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitoringReportConfig:
    """Configuration for monitoring report generation.

    Attributes:
        title: Report title.
        include_drift: Whether to include data drift section.
        include_performance: Whether to include performance section.
        include_alerts: Whether to include alerts section.
        include_concept_drift: Whether to include concept drift section.
        max_features_per_chart: Maximum features to show per chart.
        color_scheme: CSS color scheme ('light' or 'dark').
    """

    title: str = "ChurnGuard Monitoring Report"
    include_drift: bool = True
    include_performance: bool = True
    include_alerts: bool = True
    include_concept_drift: bool = True
    max_features_per_chart: int = 20
    color_scheme: str = "light"


# ---------------------------------------------------------------------------
# Chart generation (inline SVG/HTML)
# ---------------------------------------------------------------------------


def _bar_chart_html(
    labels: list[str],
    values: list[float],
    title: str,
    color: str = "#4CAF50",
    threshold_line: float | None = None,
    threshold_label: str = "",
    width: int = 600,
    height: int = 300,
) -> str:
    """Generate an inline HTML bar chart.

    Parameters
    ----------
    labels : list of str
        Bar labels.
    values : list of float
        Bar values.
    title : str
        Chart title.
    color : str
        Bar color.
    threshold_line : float, optional
        Horizontal threshold line value.
    threshold_label : str
        Label for the threshold line.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.

    Returns
    -------
    str
        HTML string.
    """
    if not labels or not values:
        return f"<p><em>No data for: {title}</em></p>"

    n = len(labels)
    max_val = max(values) if values else 1.0
    if threshold_line is not None:
        max_val = max(max_val, threshold_line * 1.2)
    max_val = max(max_val, 0.01)  # avoid division by zero

    margin_left = 120
    margin_bottom = 80
    chart_width = width - margin_left - 40
    chart_height = height - 40 - margin_bottom
    bar_height = max(chart_height / n - 4, 2)

    # Build bars
    bars_html = ""
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 20 + i * (chart_height / n)
        bar_w = (val / max_val) * chart_width

        # Color by value relative to threshold
        bar_color = color
        if threshold_line is not None:
            if val >= threshold_line:
                bar_color = "#F44336"  # red
            elif val >= threshold_line * 0.7:
                bar_color = "#FF9800"  # orange

        bars_html += f"""
        <g>
            <text x="{margin_left - 5}" y="{y + bar_height / 2 + 4}"
                  text-anchor="end" font-size="11"
                  fill="#333">{label[:25]}</text>
            <rect x="{margin_left}" y="{y}" width="{bar_w}"
                  height="{bar_height}" fill="{bar_color}" rx="2">
                <title>{label}: {val:.4f}</title>
            </rect>
            <text x="{margin_left + bar_w + 5}" y="{y + bar_height / 2 + 4}"
                  font-size="10" fill="#666">{val:.4f}</text>
        </g>"""

    # Threshold line
    threshold_html = ""
    if threshold_line is not None:
        tx = margin_left + (threshold_line / max_val) * chart_width
        threshold_html = f"""
        <line x1="{tx}" y1="15" x2="{tx}" y2="{height - margin_bottom}"
              stroke="#F44336" stroke-width="2" stroke-dasharray="5,3"/>
        <text x="{tx + 5}" y="12" font-size="10" fill="#F44336">
            {threshold_label or f"Threshold: {threshold_line:.2f}"}
        </text>"""

    return f"""
    <div class="chart-container">
        <h4>{title}</h4>
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
            {bars_html}
            {threshold_html}
        </svg>
    </div>"""


def _line_chart_html(
    series: dict[str, list[float]],
    x_labels: list[str],
    title: str,
    y_label: str = "Value",
    width: int = 600,
    height: int = 300,
) -> str:
    """Generate an inline HTML line chart.

    Parameters
    ----------
    series : dict
        Series name → list of values.
    x_labels : list of str
        X-axis labels.
    title : str
        Chart title.
    y_label : str
        Y-axis label.
    width : int
        Chart width.
    height : int
        Chart height.

    Returns
    -------
    str
        HTML string.
    """
    if not series or not x_labels:
        return f"<p><em>No data for: {title}</em></p>"

    margin_left = 60
    margin_bottom = 60
    margin_top = 30
    margin_right = 100
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    # Find global min/max
    all_vals = [v for vals in series.values() for v in vals if v is not None]
    if not all_vals:
        return f"<p><em>No data for: {title}</em></p>"

    y_min = min(all_vals)
    y_max = max(all_vals)
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    y_range = y_max - y_min

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    lines_html = ""
    legend_html = ""

    for idx, (name, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = []
        n_points = min(len(values), len(x_labels))
        for i in range(n_points):
            v = values[i]
            if v is None:
                continue
            x = margin_left + (i / max(n_points - 1, 1)) * chart_width
            y = margin_top + (1 - (v - y_min) / y_range) * chart_height
            points.append(f"{x:.1f},{y:.1f}")

        if points:
            polyline = " ".join(points)
            lines_html += f"""
            <polyline points="{polyline}" fill="none"
                      stroke="{color}" stroke-width="2"/>
            """
            # Data points
            for pt in points:
                x, y = pt.split(",")
                lines_html += f"""
                <circle cx="{x}" cy="{y}" r="3" fill="{color}">
                    <title>{name}: {values[points.index(pt)]:.4f}</title>
                </circle>"""

        legend_html += f"""
        <circle cx="{width - margin_right + 10}" cy="{margin_top + 15 + idx * 18}"
                r="5" fill="{color}"/>
        <text x="{width - margin_right + 20}" y="{margin_top + 19 + idx * 18}"
              font-size="11" fill="#333">{name}</text>"""

    # X-axis labels (show subset)
    n_labels = min(len(x_labels), 10)
    x_ticks_html = ""
    for i in range(n_labels):
        idx = int(i * (len(x_labels) - 1) / max(n_labels - 1, 1))
        x = margin_left + (idx / max(len(x_labels) - 1, 1)) * chart_width
        label = x_labels[idx][:10]  # truncate
        x_ticks_html += f"""
        <text x="{x}" y="{height - margin_bottom + 15}" font-size="9"
              fill="#666" text-anchor="middle">{label}</text>"""

    return f"""
    <div class="chart-container">
        <h4>{title}</h4>
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
            <!-- Grid -->
            <line x1="{margin_left}" y1="{margin_top}"
                  x2="{margin_left}" y2="{height - margin_bottom}"
                  stroke="#ddd" stroke-width="1"/>
            <line x1="{margin_left}" y1="{height - margin_bottom}"
                  x2="{width - margin_right}" y2="{height - margin_bottom}"
                  stroke="#ddd" stroke-width="1"/>

            <!-- Y-axis label -->
            <text x="15" y="{height / 2}" font-size="11" fill="#666"
                  transform="rotate(-90, 15, {height / 2})">{y_label}</text>

            {lines_html}
            {x_ticks_html}
            {legend_html}
        </svg>
    </div>"""


def _severity_badge(severity: DriftSeverity) -> str:
    """Generate an HTML badge for drift severity."""
    colors = {
        DriftSeverity.NONE: "#4CAF50",
        DriftSeverity.LOW: "#8BC34A",
        DriftSeverity.MEDIUM: "#FF9800",
        DriftSeverity.HIGH: "#F44336",
        DriftSeverity.CRITICAL: "#B71C1C",
    }
    color = colors.get(severity, "#999")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:3px;font-size:12px;">{severity.value.upper()}</span>'
    )


def _alert_severity_badge(severity: AlertSeverity) -> str:
    """Generate an HTML badge for alert severity."""
    colors = {
        AlertSeverity.INFO: "#2196F3",
        AlertSeverity.WARNING: "#FF9800",
        AlertSeverity.CRITICAL: "#F44336",
    }
    color = colors.get(severity, "#999")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:3px;font-size:12px;">{severity.value.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# MonitoringReport
# ---------------------------------------------------------------------------


class MonitoringReport:
    """Generate HTML monitoring reports.

    Parameters
    ----------
    config : MonitoringReportConfig, optional
        Report configuration.
    """

    def __init__(self, config: MonitoringReportConfig | None = None) -> None:
        self.config = config or MonitoringReportConfig()

    def generate(
        self,
        drift_result: DriftResult | None = None,
        performance_monitor: PerformanceMonitor | None = None,
        alert_manager: AlertManager | None = None,
        concept_drift_results: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate the HTML monitoring report.

        Parameters
        ----------
        drift_result : DriftResult, optional
            Data drift detection result.
        performance_monitor : PerformanceMonitor, optional
            Performance monitor with historical data.
        alert_manager : AlertManager, optional
            Alert manager with alert history.
        concept_drift_results : dict, optional
            Concept drift detector results.
        output_path : str or Path, optional
            Path to save the report. If None, just returns HTML.

        Returns
        -------
        str
            Complete HTML report.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cfg = self.config

        # Build sections
        sections_html = ""

        if cfg.include_drift and drift_result is not None:
            sections_html += self._build_drift_section(drift_result)

        if cfg.include_performance and performance_monitor is not None:
            sections_html += self._build_performance_section(performance_monitor)

        if cfg.include_alerts and alert_manager is not None:
            sections_html += self._build_alerts_section(alert_manager)

        if cfg.include_concept_drift and concept_drift_results is not None:
            sections_html += self._build_concept_drift_section(concept_drift_results)

        # Overall status
        overall_status = self._compute_overall_status(
            drift_result, performance_monitor, alert_manager
        )

        html = self._wrap_html(
            title=cfg.title,
            timestamp=timestamp,
            overall_status=overall_status,
            sections=sections_html,
        )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("Monitoring report saved to %s", path)

        return html

    def _compute_overall_status(
        self,
        drift_result: DriftResult | None,
        performance_monitor: PerformanceMonitor | None,
        alert_manager: AlertManager | None,
    ) -> str:
        """Compute the overall health status."""
        if alert_manager and alert_manager.unacknowledged:
            critical = any(
                a.severity == AlertSeverity.CRITICAL for a in alert_manager.unacknowledged
            )
            if critical:
                return "CRITICAL"
            return "WARNING"

        if drift_result and drift_result.overall_severity in (
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ):
            return "WARNING"

        if performance_monitor:
            trends = performance_monitor.get_trends()
            if any(t == "degrading" for t in trends.values()):
                return "WARNING"

        return "HEALTHY"

    def _build_drift_section(self, result: DriftResult) -> str:
        """Build the data drift section HTML."""
        # PSI chart
        psi_labels = [r.feature_name for r in result.psi_results]
        psi_values = [r.psi_value for r in result.psi_results]
        # Sort by PSI descending
        sorted_pairs = sorted(zip(psi_values, psi_labels), reverse=True)
        psi_values_sorted = [p[0] for p in sorted_pairs[: self.config.max_features_per_chart]]
        psi_labels_sorted = [p[1] for p in sorted_pairs[: self.config.max_features_per_chart]]

        psi_chart = _bar_chart_html(
            labels=psi_labels_sorted,
            values=psi_values_sorted,
            title="Population Stability Index (PSI) by Feature",
            color="#2196F3",
            threshold_line=0.10,
            threshold_label="PSI=0.10 (moderate)",
        )

        # KS p-value chart
        ks_labels = [r.feature_name for r in result.ks_results]
        ks_pvalues = [r.p_value for r in result.ks_results]
        # Sort by p-value ascending (most significant first)
        sorted_ks = sorted(zip(ks_pvalues, ks_labels))
        ks_values_sorted = [p[0] for p in sorted_ks[: self.config.max_features_per_chart]]
        ks_labels_sorted = [p[1] for p in sorted_ks[: self.config.max_features_per_chart]]

        ks_chart = _bar_chart_html(
            labels=ks_labels_sorted,
            values=ks_values_sorted,
            title="KS Test P-Values by Feature (lower = more drift)",
            color="#4CAF50",
            threshold_line=0.05,
            threshold_label="α=0.05",
        )

        # Summary stats
        severity_badge = _severity_badge(result.overall_severity)

        return f"""
        <section class="report-section" id="drift">
            <h2>📊 Data Drift Detection</h2>
            <div class="summary-cards">
                <div class="card">
                    <div class="card-value">{result.n_features_tested}</div>
                    <div class="card-label">Features Tested</div>
                </div>
                <div class="card">
                    <div class="card-value">{result.n_features_drifted}</div>
                    <div class="card-label">Features Drifted</div>
                </div>
                <div class="card">
                    <div class="card-value">{result.drift_score:.4f}</div>
                    <div class="card-label">Drift Score</div>
                </div>
                <div class="card">
                    <div class="card-value">{severity_badge}</div>
                    <div class="card-label">Overall Severity</div>
                </div>
            </div>
            {psi_chart}
            {ks_chart}
            <p class="footnote">Checked at {result.timestamp}</p>
        </section>"""

    def _build_performance_section(self, monitor: PerformanceMonitor) -> str:
        """Build the performance monitoring section HTML."""
        if not monitor.snapshots:
            return """
            <section class="report-section" id="performance">
                <h2>📈 Model Performance</h2>
                <p><em>No performance data available yet.</em></p>
            </section>"""

        # Build line chart for key metrics
        metrics_for_chart = ["f1", "roc_auc", "recall", "precision"]
        series: dict[str, list[float]] = {}
        for metric_name in metrics_for_chart:
            if metric_name in monitor.history:
                hist = monitor.history[metric_name]
                if len(hist) > 0:
                    series[metric_name] = list(hist.values)

        x_labels = []
        if monitor.snapshots:
            for s in monitor.snapshots:
                x_labels.append(s.timestamp[:10])  # date only

        perf_chart = _line_chart_html(
            series=series,
            x_labels=x_labels,
            title="Model Performance Over Time",
            y_label="Score",
            width=700,
            height=350,
        )

        # Latest snapshot
        latest = monitor.snapshots[-1]
        trends = monitor.get_trends()
        trends_html = ""
        for metric, trend in trends.items():
            icon = {"improving": "📈", "stable": "➡️", "degrading": "📉"}.get(trend, "?")
            trends_html += f"<span class='trend-badge'>{icon} {metric}: {trend}</span> "

        # Alerts
        alerts_html = ""
        if monitor.alerts:
            alerts_html = "<h4>Performance Alerts</h4><ul>"
            for alert in monitor.alerts[-10:]:
                alerts_html += f"<li>{alert.summary()}</li>"
            alerts_html += "</ul>"

        return f"""
        <section class="report-section" id="performance">
            <h2>📈 Model Performance</h2>
            <div class="summary-cards">
                <div class="card">
                    <div class="card-value">{latest.f1:.4f}</div>
                    <div class="card-label">F1 Score</div>
                </div>
                <div class="card">
                    <div class="card-value">{latest.roc_auc:.4f}</div>
                    <div class="card-label">ROC AUC</div>
                </div>
                <div class="card">
                    <div class="card-value">{latest.recall:.4f}</div>
                    <div class="card-label">Recall</div>
                </div>
                <div class="card">
                    <div class="card-value">{latest.precision:.4f}</div>
                    <div class="card-label">Precision</div>
                </div>
            </div>
            <div class="trends">{trends_html}</div>
            {perf_chart}
            {alerts_html}
        </section>"""

    def _build_alerts_section(self, alert_mgr: AlertManager) -> str:
        """Build the alerts section HTML."""
        alerts = alert_mgr.history[-50:]  # last 50 alerts
        if not alerts:
            return """
            <section class="report-section" id="alerts">
                <h2>🔔 Alerts</h2>
                <p><em>No alerts.</em></p>
            </section>"""

        rows_html = ""
        for alert in reversed(alerts):
            badge = _alert_severity_badge(alert.severity)
            ack_icon = "✅" if alert.acknowledged else "⬜"
            rows_html += f"""
            <tr>
                <td>{badge}</td>
                <td>{alert.timestamp[:19]}</td>
                <td>{alert.rule_name}</td>
                <td>{alert.metric_name}</td>
                <td>{alert.current_value:.4f}</td>
                <td>{alert.threshold:.4f}</td>
                <td>{alert.message}</td>
                <td>{ack_icon}</td>
            </tr>"""

        return f"""
        <section class="report-section" id="alerts">
            <h2>🔔 Alert History</h2>
            <table class="alert-table">
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Time</th>
                        <th>Rule</th>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Message</th>
                        <th>Ack</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </section>"""

    def _build_concept_drift_section(self, results: dict[str, Any]) -> str:
        """Build the concept drift section HTML."""
        content = "<p>Concept drift detection results:</p>"
        for _detector_name, result in results.items():
            if hasattr(result, "summary"):
                content += f"<p><code>{result.summary()}</code></p>"
            elif isinstance(result, dict):
                content += f"<p><code>{json.dumps(result, indent=2, default=str)}</code></p>"

        return f"""
        <section class="report-section" id="concept-drift">
            <h2>🔄 Concept Drift</h2>
            {content}
        </section>"""

    def _wrap_html(
        self,
        title: str,
        timestamp: str,
        overall_status: str,
        sections: str,
    ) -> str:
        """Wrap the report content in a full HTML document."""
        status_colors = {
            "HEALTHY": "#4CAF50",
            "WARNING": "#FF9800",
            "CRITICAL": "#F44336",
        }
        status_color = status_colors.get(overall_status, "#999")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333333;
            --muted: #666666;
            --border: #e0e0e0;
            --accent: #2196F3;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ margin: 0; font-size: 24px; }}
        .status-badge {{
            background: {status_color};
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        .timestamp {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
        .report-section {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 24px 32px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .report-section h2 {{ margin-top: 0; font-size: 20px; }}
        .summary-cards {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .card {{
            background: var(--bg);
            border-radius: 6px;
            padding: 16px 20px;
            min-width: 140px;
            text-align: center;
            flex: 1;
        }}
        .card-value {{
            font-size: 28px;
            font-weight: bold;
            color: var(--text);
        }}
        .card-label {{
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            margin-top: 4px;
        }}
        .chart-container {{
            margin: 20px 0;
            overflow-x: auto;
        }}
        .chart-container h4 {{ margin-bottom: 8px; }}
        .trends {{ margin: 12px 0; }}
        .trend-badge {{
            display: inline-block;
            background: var(--bg);
            padding: 4px 10px;
            border-radius: 4px;
            margin: 2px 4px;
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--bg); font-weight: 600; }}
        .footnote {{ color: var(--muted); font-size: 12px; margin-top: 12px; }}
        footer {{
            text-align: center;
            color: var(--muted);
            font-size: 12px;
            margin-top: 32px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>{title}</h1>
                <div class="timestamp">Generated: {timestamp}</div>
            </div>
            <span class="status-badge">{overall_status}</span>
        </header>
        {sections}
        <footer>
            Generated by <strong>ChurnGuard</strong> Monitoring &mdash;
            <a href="https://github.com/jlaportebot/churnguard">github.com/jlaportebot/churnguard</a>
        </footer>
    </div>
</body>
</html>"""
