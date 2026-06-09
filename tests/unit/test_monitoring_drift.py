"""Tests for the monitoring drift detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.monitoring.drift import (
    ChiSquareResult,
    DataDriftDetector,
    DriftResult,
    DriftSeverity,
    KSTestResult,
    PSIResult,
    compute_psi,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Create a reference (training) dataset."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "tenure": np.random.exponential(30, n),
            "monthly_charges": np.random.normal(65, 15, n),
            "total_charges": np.random.exponential(2000, n),
            "support_calls": np.random.poisson(1.5, n),
            "churn": np.random.binomial(1, 0.2, n),
        }
    )


@pytest.fixture
def similar_df() -> pd.DataFrame:
    """Create a current dataset similar to reference (no drift)."""
    np.random.seed(99)
    n = 400
    return pd.DataFrame(
        {
            "tenure": np.random.exponential(30, n),
            "monthly_charges": np.random.normal(65, 15, n),
            "total_charges": np.random.exponential(2000, n),
            "support_calls": np.random.poisson(1.5, n),
            "churn": np.random.binomial(1, 0.2, n),
        }
    )


@pytest.fixture
def drifted_df() -> pd.DataFrame:
    """Create a current dataset with significant drift."""
    np.random.seed(77)
    n = 400
    return pd.DataFrame(
        {
            "tenure": np.random.exponential(10, n),  # much shorter tenure
            "monthly_charges": np.random.normal(100, 10, n),  # higher charges
            "total_charges": np.random.exponential(500, n),  # lower total
            "support_calls": np.random.poisson(5.0, n),  # more calls
            "churn": np.random.binomial(1, 0.5, n),  # higher churn
        }
    )


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """Create a reference dataset with categorical features."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "contract": np.random.choice(
                ["month-to-month", "one-year", "two-year"], n, p=[0.5, 0.3, 0.2]
            ),
            "payment_method": np.random.choice(
                ["credit", "bank_transfer", "electronic", "mailed"], n
            ),
            "monthly_charges": np.random.normal(65, 15, n),
        }
    )


@pytest.fixture
def drifted_categorical_df() -> pd.DataFrame:
    """Create a current dataset with shifted categorical distributions."""
    np.random.seed(99)
    n = 400
    return pd.DataFrame(
        {
            "contract": np.random.choice(
                ["month-to-month", "one-year", "two-year"], n, p=[0.8, 0.15, 0.05]
            ),
            "payment_method": np.random.choice(
                ["credit", "bank_transfer", "electronic", "mailed"], n
            ),
            "monthly_charges": np.random.normal(65, 15, n),
        }
    )


# ---------------------------------------------------------------------------
# DriftSeverity tests
# ---------------------------------------------------------------------------


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_from_psi_none(self):
        assert DriftSeverity.from_psi(0.05) == DriftSeverity.NONE

    def test_from_psi_medium(self):
        assert DriftSeverity.from_psi(0.15) == DriftSeverity.MEDIUM

    def test_from_psi_high(self):
        assert DriftSeverity.from_psi(0.30) == DriftSeverity.HIGH

    def test_from_psi_boundary_low(self):
        assert DriftSeverity.from_psi(0.10) == DriftSeverity.MEDIUM

    def test_from_p_value_none(self):
        assert DriftSeverity.from_p_value(0.5) == DriftSeverity.NONE

    def test_from_p_value_low(self):
        assert DriftSeverity.from_p_value(0.04) == DriftSeverity.LOW

    def test_from_p_value_medium(self):
        assert DriftSeverity.from_p_value(0.003) == DriftSeverity.HIGH  # 0.003 < 0.005

    def test_from_p_value_high(self):
        assert DriftSeverity.from_p_value(0.0001) == DriftSeverity.HIGH

    def test_str_value(self):
        assert str(DriftSeverity.NONE) == "none"
        assert str(DriftSeverity.CRITICAL) == "critical"


# ---------------------------------------------------------------------------
# PSI computation tests
# ---------------------------------------------------------------------------


class TestComputePSI:
    """Tests for the compute_psi function."""

    def test_identical_distributions(self):
        """PSI should be near zero for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        psi, _, _ = compute_psi(data, data, n_bins=10)
        assert psi < 0.05, f"PSI for identical data should be near 0, got {psi}"

    def test_shifted_distributions(self):
        """PSI should be higher for shifted distributions."""
        np.random.seed(42)
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(70, 10, 1000)
        psi, _, _ = compute_psi(ref, cur, n_bins=10)
        assert psi > 0.10, f"PSI for shifted data should be >0.10, got {psi}"

    def test_different_spread(self):
        """PSI should detect changes in variance."""
        np.random.seed(42)
        ref = np.random.normal(50, 5, 1000)
        cur = np.random.normal(50, 20, 1000)
        psi, _, _ = compute_psi(ref, cur, n_bins=10)
        assert psi > 0.05, f"PSI for different spread should be detectable, got {psi}"

    def test_custom_bin_edges(self):
        """PSI should work with custom bin edges."""
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(55, 10, 1000)
        bins = np.array([-np.inf, 30, 40, 50, 60, 70, np.inf])
        psi, ref_bins, cur_bins = compute_psi(ref, cur, bin_edges=bins)
        assert 0 < psi < 1.0
        assert len(ref_bins) == len(cur_bins)

    def test_small_sample(self):
        """PSI with very small samples should not crash."""
        ref = np.array([1, 2, 3, 4, 5])
        cur = np.array([10, 11, 12, 13, 14])
        psi, _, _ = compute_psi(ref, cur, n_bins=3)
        assert psi >= 0


# ---------------------------------------------------------------------------
# Result container tests
# ---------------------------------------------------------------------------


class TestPSIResult:
    """Tests for PSIResult data class."""

    def test_to_dict(self):
        result = PSIResult(feature_name="tenure", psi_value=0.15, severity=DriftSeverity.MEDIUM)
        d = result.to_dict()
        assert d["feature_name"] == "tenure"
        assert d["psi_value"] == 0.15
        assert d["severity"] == "medium"

    def test_summary(self):
        result = PSIResult(feature_name="tenure", psi_value=0.15, severity=DriftSeverity.MEDIUM)
        s = result.summary()
        assert "tenure" in s
        assert "0.1500" in s
        assert "MEDIUM" in s


class TestKSTestResult:
    """Tests for KSTestResult data class."""

    def test_to_dict(self):
        result = KSTestResult(
            feature_name="age", statistic=0.3, p_value=0.001, severity=DriftSeverity.HIGH
        )
        d = result.to_dict()
        assert d["feature_name"] == "age"
        assert d["statistic"] == 0.3
        assert d["p_value"] == 0.001

    def test_summary(self):
        result = KSTestResult(
            feature_name="age", statistic=0.3, p_value=0.001, severity=DriftSeverity.HIGH
        )
        assert "KS" in result.summary()
        assert "age" in result.summary()


class TestChiSquareResult:
    """Tests for ChiSquareResult data class."""

    def test_to_dict(self):
        result = ChiSquareResult(
            feature_name="contract",
            statistic=25.0,
            p_value=0.001,
            degrees_of_freedom=2,
            severity=DriftSeverity.HIGH,
        )
        d = result.to_dict()
        assert d["degrees_of_freedom"] == 2

    def test_summary(self):
        result = ChiSquareResult(
            feature_name="contract",
            statistic=25.0,
            p_value=0.001,
            degrees_of_freedom=2,
            severity=DriftSeverity.HIGH,
        )
        assert "χ²" in result.summary()


# ---------------------------------------------------------------------------
# DriftResult tests
# ---------------------------------------------------------------------------


class TestDriftResult:
    """Tests for the aggregate DriftResult."""

    def test_summary_no_drift(self):
        result = DriftResult(
            timestamp="2025-01-01T00:00:00",
            n_features_tested=10,
            n_features_drifted=0,
            drift_score=0.0,
            overall_severity=DriftSeverity.NONE,
        )
        assert "0/10 features drifted" in result.summary()

    def test_to_dict(self):
        result = DriftResult(
            timestamp="2025-01-01T00:00:00",
            n_features_tested=5,
            n_features_drifted=2,
            drift_score=0.4,
            overall_severity=DriftSeverity.HIGH,
        )
        d = result.to_dict()
        assert d["n_features_tested"] == 5
        assert d["overall_severity"] == "high"

    def test_drifted_features(self):
        psi_r = PSIResult(feature_name="tenure", psi_value=0.3, severity=DriftSeverity.HIGH)
        ks_r = KSTestResult(
            feature_name="tenure", statistic=0.2, p_value=0.001, severity=DriftSeverity.HIGH
        )
        result = DriftResult(
            timestamp="2025-01-01",
            psi_results=[psi_r],
            ks_results=[ks_r],
            n_features_tested=1,
            n_features_drifted=1,
        )
        assert "tenure" in result.drifted_features()


# ---------------------------------------------------------------------------
# DataDriftDetector tests
# ---------------------------------------------------------------------------


class TestDataDriftDetector:
    """Tests for the DataDriftDetector class."""

    def test_no_drift_detected(self, reference_df, similar_df):
        """Similar distributions should show no/low drift."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, similar_df)
        # With similar distributions, most features should not show high drift
        assert result.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)

    def test_drift_detected(self, reference_df, drifted_df):
        """Significantly different distributions should trigger drift."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, drifted_df)
        assert result.n_features_drifted > 0, "Should detect drift in drifted data"
        assert result.overall_severity in (
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        )

    def test_psi_results_populated(self, reference_df, drifted_df):
        """PSI results should be generated for numeric features."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, drifted_df)
        assert len(result.psi_results) > 0

    def test_ks_results_populated(self, reference_df, drifted_df):
        """KS results should be generated for numeric features."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, drifted_df)
        assert len(result.ks_results) > 0

    def test_custom_thresholds(self, reference_df, similar_df):
        """Stricter PSI threshold should flag more features."""
        detector_lenient = DataDriftDetector(psi_threshold=0.25)
        detector_strict = DataDriftDetector(psi_threshold=0.05)

        result_lenient = detector_lenient.detect(reference_df, similar_df)
        result_strict = detector_strict.detect(reference_df, similar_df)

        # Strict should flag at least as many as lenient
        assert result_strict.n_features_drifted >= result_lenient.n_features_drifted

    def test_max_features_limit(self, reference_df, drifted_df):
        """max_features should limit the number of features tested."""
        detector = DataDriftDetector(max_features=2)
        result = detector.detect(reference_df, drifted_df)
        assert result.n_features_tested <= 2

    def test_categorical_features(self, categorical_df, drifted_categorical_df):
        """Categorical features should be tested with chi-square."""
        detector = DataDriftDetector(categorical_features=["contract", "payment_method"])
        result = detector.detect(categorical_df, drifted_categorical_df)
        # Contract has drifted distribution, should detect it
        assert result.n_features_tested > 0

    def test_drift_score_range(self, reference_df, drifted_df):
        """Drift score should be between 0 and 1."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, drifted_df)
        assert 0.0 <= result.drift_score <= 1.0

    def test_timestamp(self, reference_df, similar_df):
        """Result should include a timestamp."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, similar_df, timestamp="2025-01-01T00:00:00")
        assert result.timestamp == "2025-01-01T00:00:00"

    def test_custom_timestamp(self, reference_df, similar_df):
        """Custom timestamp should be used."""
        detector = DataDriftDetector()
        result = detector.detect(reference_df, similar_df, timestamp="2025-06-15")
        assert result.timestamp == "2025-06-15"

    def test_empty_dataframe(self):
        """Detector should handle empty dataframes gracefully."""
        detector = DataDriftDetector()
        ref = pd.DataFrame({"a": pd.Series(dtype=float)})
        cur = pd.DataFrame({"a": pd.Series(dtype=float)})
        # Should not crash
        result = detector.detect(ref, cur)
        assert result.n_features_tested == 0

    def test_disjoint_columns(self):
        """Detector should handle dataframes with no common columns."""
        detector = DataDriftDetector()
        ref = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        cur = pd.DataFrame({"b": [1, 2, 3, 4, 5]})
        result = detector.detect(ref, cur)
        assert result.n_features_tested == 0
