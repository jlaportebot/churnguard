"""Data drift detection for tabular features.

Detects distribution shifts between reference (training) data and
current (production) data using multiple statistical tests:

- **PSI** (Population Stability Index): Industry-standard metric for
  detecting shifts in feature distributions. Thresholds: <0.10 = none,
  0.10–0.25 = moderate, >0.25 = significant.
- **KS test** (Kolmogorov–Smirnov): Non-parametric test for continuous
  features comparing cumulative distributions.
- **Chi-square test**: For categorical features, comparing frequency
  distributions between reference and current data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    """Severity of detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_psi(cls, psi: float) -> DriftSeverity:
        """Classify severity from a PSI value."""
        if psi < 0.10:
            return cls.NONE
        elif psi < 0.25:
            return cls.MEDIUM
        else:
            return cls.HIGH

    @classmethod
    def from_p_value(cls, p_value: float, alpha: float = 0.05) -> DriftSeverity:
        """Classify severity from a statistical p-value."""
        if p_value >= alpha:
            return cls.NONE
        elif p_value >= alpha * 0.5:
            return cls.LOW
        elif p_value >= alpha * 0.1:
            return cls.MEDIUM
        else:
            return cls.HIGH


# ---------------------------------------------------------------------------
# Per-feature result containers
# ---------------------------------------------------------------------------

@dataclass
class PSIResult:
    """PSI (Population Stability Index) result for a single feature.

    Attributes:
        feature_name: Name of the feature.
        psi_value: Computed PSI value.
        severity: Drift severity based on PSI thresholds.
        reference_bins: Bin distribution for reference data.
        current_bins: Bin distribution for current data.
    """

    feature_name: str
    psi_value: float
    severity: DriftSeverity
    reference_bins: Optional[np.ndarray] = None
    current_bins: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "feature_name": self.feature_name,
            "psi_value": round(self.psi_value, 6),
            "severity": self.severity.value,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"PSI({self.feature_name})={self.psi_value:.4f} "
            f"→ {self.severity.value.upper()}"
        )


@dataclass
class KSTestResult:
    """Kolmogorov–Smirnov test result for a single feature.

    Attributes:
        feature_name: Name of the feature.
        statistic: KS test statistic.
        p_value: P-value of the test.
        severity: Drift severity based on p-value.
    """

    feature_name: str
    statistic: float
    p_value: float
    severity: DriftSeverity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "feature_name": self.feature_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "severity": self.severity.value,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"KS({self.feature_name}): stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} → {self.severity.value.upper()}"
        )


@dataclass
class ChiSquareResult:
    """Chi-square test result for a single categorical feature.

    Attributes:
        feature_name: Name of the feature.
        statistic: Chi-square test statistic.
        p_value: P-value of the test.
        degrees_of_freedom: Degrees of freedom.
        severity: Drift severity based on p-value.
    """

    feature_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    severity: DriftSeverity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "feature_name": self.feature_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "degrees_of_freedom": self.degrees_of_freedom,
            "severity": self.severity.value,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"χ²({self.feature_name}): stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, df={self.degrees_of_freedom} "
            f"→ {self.severity.value.upper()}"
        )


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Aggregate drift detection result across all features.

    Attributes:
        timestamp: ISO timestamp of the check.
        psi_results: Per-feature PSI results.
        ks_results: Per-feature KS test results.
        chi_square_results: Per-feature chi-square results.
        n_features_tested: Total number of features tested.
        n_features_drifted: Number of features showing drift.
        drift_score: Aggregate drift score (0–1).
        overall_severity: Overall drift severity.
    """

    timestamp: str
    psi_results: List[PSIResult] = field(default_factory=list)
    ks_results: List[KSTestResult] = field(default_factory=list)
    chi_square_results: List[ChiSquareResult] = field(default_factory=list)
    n_features_tested: int = 0
    n_features_drifted: int = 0
    drift_score: float = 0.0
    overall_severity: DriftSeverity = DriftSeverity.NONE

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Drift check at {self.timestamp}: "
            f"{self.n_features_drifted}/{self.n_features_tested} features drifted, "
            f"score={self.drift_score:.4f}, "
            f"severity={self.overall_severity.value.upper()}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "n_features_tested": self.n_features_tested,
            "n_features_drifted": self.n_features_drifted,
            "drift_score": round(self.drift_score, 6),
            "overall_severity": self.overall_severity.value,
            "psi_results": [r.to_dict() for r in self.psi_results],
            "ks_results": [r.to_dict() for r in self.ks_results],
            "chi_square_results": [r.to_dict() for r in self.chi_square_results],
        }

    def drifted_features(self) -> List[str]:
        """Return feature names that show drift."""
        drifted = set()
        for r in self.psi_results:
            if r.severity != DriftSeverity.NONE:
                drifted.add(r.feature_name)
        for r in self.ks_results:
            if r.severity != DriftSeverity.NONE:
                drifted.add(r.feature_name)
        for r in self.chi_square_results:
            if r.severity != DriftSeverity.NONE:
                drifted.add(r.feature_name)
        return sorted(drifted)


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    bin_edges: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the Population Stability Index (PSI).

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution values.
    current : np.ndarray
        Current distribution values.
    n_bins : int
        Number of bins for discretization (default 10).
    bin_edges : np.ndarray, optional
        Pre-defined bin edges. If None, computed from reference data.

    Returns
    -------
    tuple of (float, np.ndarray, np.ndarray)
        PSI value, reference bin frequencies, current bin frequencies.
    """
    if bin_edges is None:
        # Use quantile-based binning from reference data
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(reference, percentiles)
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0, np.array([1.0]), np.array([1.0])
        # Extend edges to capture all data
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions
    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    # Add small epsilon to avoid log(0)
    eps = 1e-6
    ref_props = np.clip(ref_props, eps, 1.0)
    cur_props = np.clip(cur_props, eps, 1.0)

    # PSI = sum((cur - ref) * ln(cur / ref))
    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))

    return psi, ref_props, cur_props


# ---------------------------------------------------------------------------
# DataDriftDetector
# ---------------------------------------------------------------------------

class DataDriftDetector:
    """Detect data drift between reference and current datasets.

    Applies multiple statistical tests to each feature and produces
    an aggregate drift assessment.

    Parameters
    ----------
    psi_threshold : float
        PSI value above which drift is flagged (default 0.10).
    ks_alpha : float
        Significance level for KS test (default 0.05).
    chi_alpha : float
        Significance level for chi-square test (default 0.05).
    n_bins : int
        Number of bins for PSI computation (default 10).
    max_features : int, optional
        Maximum number of features to test. None = all.
    categorical_features : list of str, optional
        Feature names that should be treated as categorical.
    """

    def __init__(
        self,
        psi_threshold: float = 0.10,
        ks_alpha: float = 0.05,
        chi_alpha: float = 0.05,
        n_bins: int = 10,
        max_features: Optional[int] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.chi_alpha = chi_alpha
        self.n_bins = n_bins
        self.max_features = max_features
        self.categorical_features = set(categorical_features or [])

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        timestamp: Optional[str] = None,
    ) -> DriftResult:
        """Run drift detection on all common features.

        Parameters
        ----------
        reference : pd.DataFrame
            Reference (baseline) dataset.
        current : pd.DataFrame
            Current (production) dataset.
        timestamp : str, optional
            ISO timestamp. Defaults to now.

        Returns
        -------
        DriftResult
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Find common numeric columns
        common_cols = list(set(reference.columns) & set(current.columns))
        numeric_cols = [
            c for c in common_cols
            if pd.api.types.is_numeric_dtype(reference[c])
        ]
        categorical_cols = [
            c for c in common_cols
            if c in self.categorical_features
            or isinstance(reference[c].dtype, pd.CategoricalDtype)
            or reference[c].dtype == object
        ]

        # Limit features if needed
        if self.max_features is not None:
            numeric_cols = numeric_cols[: self.max_features]
            categorical_cols = categorical_cols[: max(0, self.max_features - len(numeric_cols))]

        psi_results: List[PSIResult] = []
        ks_results: List[KSTestResult] = []
        chi_results: List[ChiSquareResult] = []

        # --- Numeric features: PSI + KS ---
        for col in numeric_cols:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            # PSI
            try:
                psi_val, ref_bins, cur_bins = compute_psi(
                    ref_vals, cur_vals, n_bins=self.n_bins
                )
                psi_severity = DriftSeverity.from_psi(psi_val)
                psi_results.append(PSIResult(
                    feature_name=col,
                    psi_value=psi_val,
                    severity=psi_severity,
                    reference_bins=ref_bins,
                    current_bins=cur_bins,
                ))
            except Exception as e:
                logger.warning("PSI computation failed for %s: %s", col, e)

            # KS test
            try:
                ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
                ks_severity = DriftSeverity.from_p_value(ks_p, self.ks_alpha)
                ks_results.append(KSTestResult(
                    feature_name=col,
                    statistic=float(ks_stat),
                    p_value=float(ks_p),
                    severity=ks_severity,
                ))
            except Exception as e:
                logger.warning("KS test failed for %s: %s", col, e)

        # --- Categorical features: Chi-square ---
        for col in categorical_cols:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            try:
                # Build contingency table
                all_categories = sorted(set(ref_vals) | set(cur_vals))
                ref_counts = np.array([
                    np.sum(ref_vals == cat) for cat in all_categories
                ])
                cur_counts = np.array([
                    np.sum(cur_vals == cat) for cat in all_categories
                ])

                # Filter out categories with 0 in both
                mask = (ref_counts + cur_counts) > 0
                ref_counts = ref_counts[mask]
                cur_counts = cur_counts[mask]

                if len(ref_counts) < 2:
                    continue

                chi_stat, chi_p = stats.chisquare(cur_counts, f_exp=ref_counts)
                chi_severity = DriftSeverity.from_p_value(chi_p, self.chi_alpha)

                chi_results.append(ChiSquareResult(
                    feature_name=col,
                    statistic=float(chi_stat),
                    p_value=float(chi_p),
                    degrees_of_freedom=int(len(ref_counts) - 1),
                    severity=chi_severity,
                ))
            except Exception as e:
                logger.warning("Chi-square test failed for %s: %s", col, e)

        # Aggregate results
        n_tested = len(psi_results) + len(chi_results)
        n_drifted = sum(
            1 for r in psi_results if r.severity != DriftSeverity.NONE
        ) + sum(
            1 for r in ks_results if r.severity != DriftSeverity.NONE
        ) + sum(
            1 for r in chi_results if r.severity != DriftSeverity.NONE
        )

        # Compute aggregate drift score (0–1)
        if n_tested > 0:
            drift_score = min(n_drifted / n_tested, 1.0)
        else:
            drift_score = 0.0

        # Overall severity
        if drift_score == 0:
            overall_severity = DriftSeverity.NONE
        elif drift_score < 0.15:
            overall_severity = DriftSeverity.LOW
        elif drift_score < 0.40:
            overall_severity = DriftSeverity.MEDIUM
        elif drift_score < 0.70:
            overall_severity = DriftSeverity.HIGH
        else:
            overall_severity = DriftSeverity.CRITICAL

        return DriftResult(
            timestamp=timestamp,
            psi_results=psi_results,
            ks_results=ks_results,
            chi_square_results=chi_results,
            n_features_tested=n_tested,
            n_features_drifted=n_drifted,
            drift_score=drift_score,
            overall_severity=overall_severity,
        )
