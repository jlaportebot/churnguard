"""Concept drift detection algorithms for streaming data.

Provides implementations of well-known concept drift detection methods:

- **ADWIN** (ADaptive WINdowing): Detects change by adaptively shrinking
  a sliding window when two sub-windows have significantly different means.
- **DDM** (Drift Detection Method): Monitors error rate and standard
  deviation, triggers drift when error exceeds a threshold.
- **EDDM** (Early Drift Detection Method): Monitors distance between
  classification errors rather than error rate, providing earlier detection.

These are designed for streaming / online scenarios where predictions
arrive one at a time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class DriftState(str, Enum):
    """State of the concept drift detector."""

    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"

    def __str__(self) -> str:
        return self.value


@dataclass
class ConceptDriftResult:
    """Result from a concept drift detection check.

    Attributes:
        state: Current drift state (stable, warning, drift).
        detector_name: Name of the detector algorithm.
        statistic: The detection statistic value.
        threshold: The threshold that was compared against.
        n_samples: Number of samples processed so far.
        n_warnings: Number of warnings issued.
        n_drifts: Number of drifts detected.
        details: Algorithm-specific details.
    """

    state: DriftState
    detector_name: str
    statistic: float = 0.0
    threshold: float = 0.0
    n_samples: int = 0
    n_warnings: int = 0
    n_drifts: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.detector_name}] State: {self.state.value.upper()}, "
            f"statistic={self.statistic:.4f}, threshold={self.threshold:.4f}, "
            f"samples={self.n_samples}, warnings={self.n_warnings}, "
            f"drifts={self.n_drifts}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "state": self.state.value,
            "detector_name": self.detector_name,
            "statistic": round(self.statistic, 6),
            "threshold": round(self.threshold, 6),
            "n_samples": self.n_samples,
            "n_warnings": self.n_warnings,
            "n_drifts": self.n_drifts,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ConceptDriftDetector(ABC):
    """Abstract base class for concept drift detectors.

    All detectors follow a streaming interface:
    1. Call :meth:`update` with each new prediction/label pair.
    2. Check :meth:`detect` for the current drift state.
    3. Call :meth:`reset` to restart the detector.
    """

    @abstractmethod
    def update(self, prediction: int, label: int) -> ConceptDriftResult:
        """Process a new prediction-label pair.

        Parameters
        ----------
        prediction : int
            Predicted label (0 or 1).
        label : int
            True label (0 or 1).

        Returns
        -------
        ConceptDriftResult
        """

    @abstractmethod
    def detect(self) -> ConceptDriftResult:
        """Return the current drift state without updating.

        Returns
        -------
        ConceptDriftResult
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector to its initial state."""

    def update_batch(self, predictions: np.ndarray, labels: np.ndarray) -> list[ConceptDriftResult]:
        """Process a batch of prediction-label pairs.

        Parameters
        ----------
        predictions : np.ndarray
            Array of predicted labels.
        labels : np.ndarray
            Array of true labels.

        Returns
        -------
        list of ConceptDriftResult
        """
        results = []
        for pred, label in zip(predictions, labels):
            result = self.update(int(pred), int(label))
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# ADWIN (Adaptive Windowing)
# ---------------------------------------------------------------------------


class ADWIN(ConceptDriftDetector):
    """ADaptive WINdowing drift detector.

    Maintains a variable-length window and checks whether splitting
    the window into two sub-windows yields significantly different
    means (using the Hoeffding bound). When drift is detected, the
    older sub-window is dropped.

    Parameters
    ----------
    delta : float
        Confidence parameter (default 0.002). Smaller values make
        the detector less sensitive (fewer false alarms).
    min_window_size : int
        Minimum number of samples before drift can be detected.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window_size: int = 5,
    ) -> None:
        self.delta = delta
        self.min_window_size = min_window_size
        self._window: list[float] = []
        self._n_samples = 0
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE
        self._total_removed = 0

    def update(self, prediction: int, label: int) -> ConceptDriftResult:
        """Process a new prediction-label pair.

        The error indicator (1 if wrong, 0 if correct) is added to
        the window, and the window is checked for drift.
        """
        error = 0.0 if prediction == label else 1.0
        self._window.append(error)
        self._n_samples += 1

        # Check for drift
        drift_detected = self._check_drift()

        if drift_detected:
            self._n_drifts += 1
            self._last_state = DriftState.DRIFT
            # Remove elements from the older sub-window
            # (the one with the higher mean)
            n = len(self._window)
            # Try all possible cut points to find the best split
            # where the difference is maximal
            best_cut = 0
            best_diff = 0.0
            for cut in range(self.min_window_size, n - self.min_window_size + 1):
                mean_0 = np.mean(self._window[:cut])
                mean_1 = np.mean(self._window[cut:])
                diff = abs(mean_0 - mean_1)
                if diff > best_diff:
                    best_diff = diff
                    best_cut = cut

            # Remove older portion
            removed = self._window[:best_cut]
            self._total_removed += len(removed)
            self._window = self._window[best_cut:]
        else:
            # Check for warning state
            if len(self._window) >= 2 * self.min_window_size:
                n = len(self._window)
                mid = n // 2
                mean_0 = np.mean(self._window[:mid])
                mean_1 = np.mean(self._window[mid:])
                epsilon = self._hoeffding_bound(self.delta * 2, mid, n - mid)
                if abs(mean_0 - mean_1) > epsilon * 0.7:
                    self._n_warnings += 1
                    self._last_state = DriftState.WARNING
                else:
                    self._last_state = DriftState.STABLE
            else:
                self._last_state = DriftState.STABLE

        return self.detect()

    def detect(self) -> ConceptDriftResult:
        """Return the current drift state."""
        window_mean = float(np.mean(self._window)) if self._window else 0.0
        return ConceptDriftResult(
            state=self._last_state,
            detector_name="ADWIN",
            statistic=window_mean,
            threshold=self.delta,
            n_samples=self._n_samples,
            n_warnings=self._n_warnings,
            n_drifts=self._n_drifts,
            details={
                "window_size": len(self._window),
                "window_mean": window_mean,
                "delta": self.delta,
                "total_removed": self._total_removed,
            },
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._window = []
        self._n_samples = 0
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE
        self._total_removed = 0

    def _check_drift(self) -> bool:
        """Check if drift is detected in the current window."""
        n = len(self._window)
        if n < 2 * self.min_window_size:
            return False

        # Check all possible cut points
        for cut in range(self.min_window_size, n - self.min_window_size + 1):
            mean_0 = np.mean(self._window[:cut])
            mean_1 = np.mean(self._window[cut:])
            n_0 = cut
            n_1 = n - cut

            epsilon = self._hoeffding_bound(self.delta, n_0, n_1)

            if abs(mean_0 - mean_1) >= epsilon:
                return True

        return False

    @staticmethod
    def _hoeffding_bound(delta: float, n_0: int, n_1: int) -> float:
        """Compute the Hoeffding bound for two sub-window sizes.

        The bound is:

            ε = sqrt(0.5 * m * ln(2/δ))

        where m = 1/n_0 + 1/n_1.
        """
        m = 1.0 / n_0 + 1.0 / n_1
        return float(np.sqrt(0.5 * m * np.log(2.0 / delta)))

    @property
    def window_size(self) -> int:
        """Current window size."""
        return len(self._window)

    @property
    def window_mean(self) -> float:
        """Current mean of the window."""
        return float(np.mean(self._window)) if self._window else 0.0


# ---------------------------------------------------------------------------
# DDM (Drift Detection Method)
# ---------------------------------------------------------------------------


class DDM(ConceptDriftDetector):
    """Drift Detection Method.

    Monitors the error rate of a classifier. As samples arrive,
    it tracks the running error rate (p) and standard deviation (s).

    - **Warning**: p + s ≥ p_min + 2 * s_min
    - **Drift**: p + s ≥ p_min + 3 * s_min

    Where p_min and s_min are the minimum observed values of p and s.

    Parameters
    ----------
    warning_level : float
        Number of standard deviations for warning (default 2.0).
    drift_level : float
        Number of standard deviations for drift (default 3.0).
    min_samples : int
        Minimum samples before detection starts.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30,
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self._n_samples = 0
        self._n_errors = 0
        self._p_min: float | None = None
        self._s_min: float | None = None
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE

    def update(self, prediction: int, label: int) -> ConceptDriftResult:
        """Process a new prediction-label pair."""
        self._n_samples += 1
        if prediction != label:
            self._n_errors += 1

        if self._n_samples < self.min_samples:
            self._last_state = DriftState.STABLE
            return self.detect()

        # Current error rate and standard deviation
        p = self._n_errors / self._n_samples
        s = np.sqrt(p * (1 - p) / self._n_samples)

        # Track minimum p + s
        if self._p_min is None or (p + s) < (self._p_min + self._s_min):
            self._p_min = p
            self._s_min = s

        # Check levels
        if self._p_min is not None and self._s_min is not None:
            if p + s >= self._p_min + self.drift_level * self._s_min:
                self._n_drifts += 1
                self._last_state = DriftState.DRIFT
                # Reset after drift detection
                self._reset_stats()
            elif p + s >= self._p_min + self.warning_level * self._s_min:
                self._n_warnings += 1
                self._last_state = DriftState.WARNING
            else:
                self._last_state = DriftState.STABLE

        return self.detect()

    def detect(self) -> ConceptDriftResult:
        """Return the current drift state."""
        p = self._n_errors / max(self._n_samples, 1)
        return ConceptDriftResult(
            state=self._last_state,
            detector_name="DDM",
            statistic=float(p),
            threshold=float(self._p_min + self.drift_level * self._s_min)
            if self._p_min is not None and self._s_min is not None
            else 0.0,
            n_samples=self._n_samples,
            n_warnings=self._n_warnings,
            n_drifts=self._n_drifts,
            details={
                "error_rate": round(float(p), 6),
                "p_min": round(float(self._p_min), 6) if self._p_min else None,
                "s_min": round(float(self._s_min), 6) if self._s_min else None,
                "warning_level": self.warning_level,
                "drift_level": self.drift_level,
            },
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._n_samples = 0
        self._n_errors = 0
        self._p_min = None
        self._s_min = None
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE

    def _reset_stats(self) -> None:
        """Reset tracking statistics after drift (keeps sample count)."""
        self._n_errors = 0
        self._n_samples = 0
        self._p_min = None
        self._s_min = None


# ---------------------------------------------------------------------------
# EDDM (Early Drift Detection Method)
# ---------------------------------------------------------------------------


class EDDM(ConceptDriftDetector):
    """Early Drift Detection Method.

    Instead of monitoring error rate, EDDM monitors the distance
    between consecutive classification errors. When the mean distance
    between errors drops significantly, drift is signaled.

    - **Warning**: p'_max + 2 * s'_max < p' + 2 * s'
    - **Drift**: p'_max + 2 * s'_max < p' + 3 * s'

    Where p' is the mean distance between errors and s' is the
    standard deviation.

    Parameters
    ----------
    alpha : float
        Scaling factor for the warning threshold (default 0.95).
    beta : float
        Scaling factor for the drift threshold (default 0.90).
    min_samples : int
        Minimum number of errors before detection starts.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        beta: float = 0.90,
        min_samples: int = 30,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.min_samples = min_samples
        self._n_samples = 0
        self._n_errors = 0
        self._last_error_index: int | None = None
        self._distances: list[float] = []
        self._p_prime_max: float | None = None
        self._s_prime_max: float | None = None
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE

    def update(self, prediction: int, label: int) -> ConceptDriftResult:
        """Process a new prediction-label pair."""
        self._n_samples += 1

        if prediction != label:
            self._n_errors += 1

            if self._last_error_index is not None:
                distance = float(self._n_samples - self._last_error_index)
                self._distances.append(distance)

            self._last_error_index = self._n_samples

            # Need at least min_samples errors with distances
            if self._n_errors >= self.min_samples and len(self._distances) >= 2:
                self._check_eddm()

        return self.detect()

    def _check_eddm(self) -> None:
        """Check EDDM conditions."""
        distances = np.array(self._distances)
        p_prime = float(np.mean(distances))
        s_prime = float(np.std(distances))

        # Track maximum of (p' + 2 * s')
        current = p_prime + 2 * s_prime

        if self._p_prime_max is None or current > (self._p_prime_max + 2 * self._s_prime_max):
            self._p_prime_max = p_prime
            self._s_prime_max = s_prime

        if self._p_prime_max is not None and self._s_prime_max is not None:
            max_val = self._p_prime_max + 2 * self._s_prime_max

            # Drift: (p'_max + 2*s'_max) * beta > (p' + 2*s')
            if max_val * self.beta > current:
                self._n_drifts += 1
                self._last_state = DriftState.DRIFT
                # Reset after drift
                self._reset_stats()
            # Warning: (p'_max + 2*s'_max) * alpha > (p' + 2*s')
            elif max_val * self.alpha > current:
                self._n_warnings += 1
                self._last_state = DriftState.WARNING
            else:
                self._last_state = DriftState.STABLE

    def detect(self) -> ConceptDriftResult:
        """Return the current drift state."""
        if self._distances:
            p_prime = float(np.mean(self._distances))
            s_prime = float(np.std(self._distances))
            stat = p_prime + 2 * s_prime
        else:
            stat = 0.0

        threshold = 0.0
        if self._p_prime_max is not None and self._s_prime_max is not None:
            threshold = (self._p_prime_max + 2 * self._s_prime_max) * self.beta

        return ConceptDriftResult(
            state=self._last_state,
            detector_name="EDDM",
            statistic=round(float(stat), 6),
            threshold=round(float(threshold), 6),
            n_samples=self._n_samples,
            n_warnings=self._n_warnings,
            n_drifts=self._n_drifts,
            details={
                "n_errors": self._n_errors,
                "mean_distance": round(float(np.mean(self._distances)), 4)
                if self._distances
                else None,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._n_samples = 0
        self._n_errors = 0
        self._last_error_index = None
        self._distances = []
        self._p_prime_max = None
        self._s_prime_max = None
        self._n_warnings = 0
        self._n_drifts = 0
        self._last_state = DriftState.STABLE

    def _reset_stats(self) -> None:
        """Reset tracking after drift (keep sample count for continuity)."""
        self._n_errors = 0
        self._last_error_index = None
        self._distances = []
        self._p_prime_max = None
        self._s_prime_max = None
