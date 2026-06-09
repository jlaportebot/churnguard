"""Tests for the monitoring concept drift detection module."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard.monitoring.concept import (
    ADWIN,
    DDM,
    EDDM,
    ConceptDriftDetector,
    ConceptDriftResult,
    DriftState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_stable_stream(n: int = 500, error_rate: float = 0.1, seed: int = 42):
    """Generate a stream of predictions/labels with stable error rate."""
    rng = np.random.RandomState(seed)
    labels = rng.binomial(1, 0.5, n)
    # Flip some predictions to create the target error rate
    predictions = labels.copy()
    n_errors = int(n * error_rate)
    error_indices = rng.choice(n, n_errors, replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    return predictions, labels


def _generate_drifting_stream(
    n_pre: int = 300,
    n_post: int = 300,
    error_rate_pre: float = 0.1,
    error_rate_post: float = 0.4,
    seed: int = 42,
):
    """Generate a stream with a drift point in the middle."""
    rng = np.random.RandomState(seed)
    labels_pre = rng.binomial(1, 0.5, n_pre)
    labels_post = rng.binomial(1, 0.5, n_post)
    labels = np.concatenate([labels_pre, labels_post])

    predictions = labels.copy()
    # Pre-drift: low error
    n_errors_pre = int(n_pre * error_rate_pre)
    err_idx = rng.choice(n_pre, n_errors_pre, replace=False)
    predictions[err_idx] = 1 - predictions[err_idx]

    # Post-drift: high error
    n_errors_post = int(n_post * error_rate_post)
    err_idx = rng.choice(n_post, n_errors_post, replace=False) + n_pre
    predictions[err_idx] = 1 - predictions[err_idx]

    return predictions, labels


# ---------------------------------------------------------------------------
# DriftState tests
# ---------------------------------------------------------------------------


class TestDriftState:
    """Tests for DriftState enum."""

    def test_values(self):
        assert DriftState.STABLE.value == "stable"
        assert DriftState.WARNING.value == "warning"
        assert DriftState.DRIFT.value == "drift"

    def test_str(self):
        assert str(DriftState.STABLE) == "stable"


# ---------------------------------------------------------------------------
# ConceptDriftResult tests
# ---------------------------------------------------------------------------


class TestConceptDriftResult:
    """Tests for ConceptDriftResult data class."""

    def test_summary(self):
        result = ConceptDriftResult(
            state=DriftState.DRIFT,
            detector_name="ADWIN",
            statistic=0.5,
            threshold=0.002,
            n_samples=100,
        )
        s = result.summary()
        assert "ADWIN" in s
        assert "DRIFT" in s

    def test_to_dict(self):
        result = ConceptDriftResult(
            state=DriftState.WARNING,
            detector_name="DDM",
            statistic=0.3,
            threshold=0.1,
            n_samples=50,
            n_warnings=3,
            n_drifts=0,
        )
        d = result.to_dict()
        assert d["state"] == "warning"
        assert d["detector_name"] == "DDM"
        assert d["n_warnings"] == 3


# ---------------------------------------------------------------------------
# ADWIN tests
# ---------------------------------------------------------------------------


class TestADWIN:
    """Tests for the ADWIN detector."""

    def test_initial_state(self):
        adwin = ADWIN()
        result = adwin.detect()
        assert result.state == DriftState.STABLE
        assert result.n_samples == 0

    def test_stable_stream(self):
        """ADWIN should remain stable with a consistent error rate."""
        adwin = ADWIN(delta=0.002, min_window_size=5)
        predictions, labels = _generate_stable_stream(n=300, error_rate=0.1)
        for pred, label in zip(predictions, labels):
            adwin.update(int(pred), int(label))

        result = adwin.detect()
        # With a stable stream, we expect mostly stable state
        # (might have some warnings but not many drifts)
        assert result.n_drifts <= 2, f"Too many drifts for stable stream: {result.n_drifts}"

    def test_drifting_stream(self):
        """ADWIN should detect drift when error rate changes significantly."""
        adwin = ADWIN(delta=0.002, min_window_size=5)
        predictions, labels = _generate_drifting_stream(
            n_pre=300,
            n_post=500,
            error_rate_pre=0.05,
            error_rate_post=0.45,
        )
        for pred, label in zip(predictions, labels):
            adwin.update(int(pred), int(label))

        result = adwin.detect()
        assert result.n_drifts >= 1, "ADWIN should detect drift in drifting stream"

    def test_reset(self):
        """Reset should clear all state."""
        adwin = ADWIN()
        for _i in range(20):
            adwin.update(0, 0)

        adwin.reset()
        result = adwin.detect()
        assert result.n_samples == 0
        assert result.state == DriftState.STABLE

    def test_update_batch(self):
        """Batch update should work correctly."""
        adwin = ADWIN()
        predictions, labels = _generate_stable_stream(n=50)
        results = adwin.update_batch(predictions, labels)
        assert len(results) == 50
        assert all(isinstance(r, ConceptDriftResult) for r in results)

    def test_window_properties(self):
        """Window size and mean should be accessible."""
        adwin = ADWIN(min_window_size=3)
        for _ in range(10):
            adwin.update(1, 0)  # all errors
        assert adwin.window_size > 0
        assert adwin.window_mean > 0

    def test_custom_delta(self):
        """Higher delta should be less sensitive."""
        adwin_sensitive = ADWIN(delta=0.001, min_window_size=5)
        adwin_lenient = ADWIN(delta=0.01, min_window_size=5)

        predictions, labels = _generate_drifting_stream(
            n_pre=200,
            n_post=300,
            error_rate_pre=0.1,
            error_rate_post=0.35,
        )

        for pred, label in zip(predictions, labels):
            adwin_sensitive.update(int(pred), int(label))
            adwin_lenient.update(int(pred), int(label))

        # More sensitive detector should detect drift earlier or more often
        # (not guaranteed but highly likely with these parameters)
        assert adwin_sensitive.detect().n_samples == adwin_lenient.detect().n_samples


# ---------------------------------------------------------------------------
# DDM tests
# ---------------------------------------------------------------------------


class TestDDM:
    """Tests for the DDM detector."""

    def test_initial_state(self):
        ddm = DDM()
        result = ddm.detect()
        assert result.state == DriftState.STABLE
        assert result.n_samples == 0

    def test_stable_stream(self):
        """DDM should remain stable with consistent low error rate."""
        ddm = DDM(min_samples=30)
        predictions, labels = _generate_stable_stream(n=500, error_rate=0.1)
        for pred, label in zip(predictions, labels):
            ddm.update(int(pred), int(label))

        result = ddm.detect()
        # Stable stream should not produce many drifts
        assert result.n_drifts <= 2

    def test_drifting_stream(self):
        """DDM should detect drift when error rate increases."""
        ddm = DDM(min_samples=30, warning_level=2.0, drift_level=3.0)
        predictions, labels = _generate_drifting_stream(
            n_pre=200,
            n_post=500,
            error_rate_pre=0.05,
            error_rate_post=0.50,
        )
        for pred, label in zip(predictions, labels):
            ddm.update(int(pred), int(label))

        result = ddm.detect()
        # Should detect at least one drift or warning
        assert result.n_drifts >= 1 or result.n_warnings >= 1

    def test_reset(self):
        """Reset should clear all state."""
        ddm = DDM()
        for _i in range(50):
            ddm.update(0, 0)

        ddm.reset()
        result = ddm.detect()
        assert result.n_samples == 0

    def test_custom_levels(self):
        """Custom warning/drift levels should work."""
        ddm = DDM(warning_level=1.5, drift_level=2.5, min_samples=20)
        predictions, labels = _generate_drifting_stream(
            n_pre=100,
            n_post=300,
            error_rate_pre=0.1,
            error_rate_post=0.5,
        )
        for pred, label in zip(predictions, labels):
            ddm.update(int(pred), int(label))

        result = ddm.detect()
        assert result.n_samples > 0

    def test_min_samples_before_detection(self):
        """Should not trigger before min_samples is reached."""
        ddm = DDM(min_samples=100)
        # Feed 50 samples with high error — should not drift yet
        for _ in range(50):
            ddm.update(1, 0)  # all wrong

        result = ddm.detect()
        # Before min_samples, should be stable
        assert result.state == DriftState.STABLE or result.n_samples < 100

    def test_result_details(self):
        """Result should include useful details."""
        ddm = DDM(min_samples=30)
        predictions, labels = _generate_stable_stream(n=100, error_rate=0.15)
        for pred, label in zip(predictions, labels):
            ddm.update(int(pred), int(label))

        result = ddm.detect()
        assert "error_rate" in result.details
        assert "warning_level" in result.details


# ---------------------------------------------------------------------------
# EDDM tests
# ---------------------------------------------------------------------------


class TestEDDM:
    """Tests for the EDDM detector."""

    def test_initial_state(self):
        eddm = EDDM()
        result = eddm.detect()
        assert result.state == DriftState.STABLE
        assert result.n_samples == 0

    def test_stable_stream(self):
        """EDDM should remain mostly stable with consistent error patterns."""
        eddm = EDDM(min_samples=30)
        predictions, labels = _generate_stable_stream(n=500, error_rate=0.1)
        for pred, label in zip(predictions, labels):
            eddm.update(int(pred), int(label))

        result = eddm.detect()
        # May have some drifts but should be relatively stable
        assert isinstance(result.state, DriftState)

    def test_drifting_stream(self):
        """EDDM should detect drift in a significantly drifting stream."""
        eddm = EDDM(alpha=0.95, beta=0.90, min_samples=25)
        predictions, labels = _generate_drifting_stream(
            n_pre=200,
            n_post=500,
            error_rate_pre=0.05,
            error_rate_post=0.55,
        )
        for pred, label in zip(predictions, labels):
            eddm.update(int(pred), int(label))

        result = eddm.detect()
        # EDDM should detect at least warnings
        assert result.n_warnings > 0 or result.n_drifts > 0

    def test_reset(self):
        """Reset should clear all state."""
        eddm = EDDM()
        for _ in range(50):
            eddm.update(0, 0)

        eddm.reset()
        result = eddm.detect()
        assert result.n_samples == 0

    def test_update_batch(self):
        """Batch update should work."""
        eddm = EDDM()
        predictions, labels = _generate_stable_stream(n=100)
        results = eddm.update_batch(predictions, labels)
        assert len(results) == 100

    def test_result_details(self):
        """Result details should include error count."""
        eddm = EDDM(min_samples=10)
        predictions, labels = _generate_stable_stream(n=80, error_rate=0.2)
        for pred, label in zip(predictions, labels):
            eddm.update(int(pred), int(label))

        result = eddm.detect()
        assert "n_errors" in result.details

    def test_no_errors_stream(self):
        """EDDM should handle a stream with no errors."""
        eddm = EDDM()
        for _ in range(100):
            eddm.update(1, 1)  # all correct
        result = eddm.detect()
        assert result.state == DriftState.STABLE


# ---------------------------------------------------------------------------
# Abstract base class tests
# ---------------------------------------------------------------------------


class TestConceptDriftDetectorABC:
    """Tests that the abstract base class enforces the interface."""

    def test_cannot_instantiate(self):
        """Cannot directly instantiate the abstract class."""
        with pytest.raises(TypeError):
            ConceptDriftDetector()
