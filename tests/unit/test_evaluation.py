"""Tests for the evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnguard.evaluation import ModelEvaluator, format_results_table
from churnguard.models.base import ModelResult


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample ModelResult for testing."""
        return ModelResult(
            model_name="TestModel",
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.90,
            pr_auc=0.82,
            confusion_matrix=np.array([[80, 10], [5, 25]]),
            feature_importance={"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2},
            y_pred=np.array([0] * 80 + [1] * 10 + [1] * 25 + [0] * 5)[:120],
            y_proba=np.random.RandomState(42).random(120),
            training_time_seconds=1.5,
            cv_scores=[0.75, 0.78, 0.76, 0.79, 0.74],
        )

    @pytest.fixture
    def y_test(self):
        """Create test target values."""
        return pd.Series([0] * 85 + [1] * 35)

    def test_evaluate_model_returns_report(self, sample_result, y_test):
        """Test that evaluate_model returns a comprehensive report."""
        evaluator = ModelEvaluator(save_plots=False, save_json=False)
        report = evaluator.evaluate_model(sample_result, y_test)
        assert "model_name" in report
        assert "metrics" in report
        assert report["metrics"]["accuracy"] == 0.85

    def test_evaluate_model_saves_json(self, sample_result, y_test, tmp_path):
        """Test that evaluate_model saves JSON report."""
        evaluator = ModelEvaluator(
            output_dir=tmp_path, save_plots=False, save_json=True
        )
        evaluator.evaluate_model(sample_result, y_test)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["model_name"] == "TestModel"

    def test_risk_distribution(self, sample_result, y_test):
        """Test risk distribution computation."""
        evaluator = ModelEvaluator(save_plots=False, save_json=False)
        report = evaluator.evaluate_model(sample_result, y_test)
        assert "risk_distribution" in report
        dist = report["risk_distribution"]
        assert "very_high_75_100" in dist
        assert "low_0_25" in dist

    def test_compare_models(self, sample_result, y_test):
        """Test model comparison."""
        results = {
            "model_a": sample_result,
            "model_b": ModelResult(
                model_name="ModelB",
                accuracy=0.80,
                f1=0.72,
                roc_auc=0.85,
            ),
        }
        evaluator = ModelEvaluator(save_plots=False, save_json=False)
        df = evaluator.compare_models(results, y_test)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output = tmp_path / "new_dir" / "sub"
        evaluator = ModelEvaluator(output_dir=output, save_plots=True)
        assert output.exists()

    def test_confusion_matrix_in_report(self, sample_result, y_test):
        """Test confusion matrix is included in report."""
        evaluator = ModelEvaluator(save_plots=False, save_json=False)
        report = evaluator.evaluate_model(sample_result, y_test)
        assert report["confusion_matrix"] is not None
        assert len(report["confusion_matrix"]) == 2


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    def test_basic_formatting(self):
        """Test basic results table formatting."""
        results = {
            "model_a": ModelResult(model_name="ModelA", f1=0.80, roc_auc=0.85, accuracy=0.85),
            "model_b": ModelResult(model_name="ModelB", f1=0.75, roc_auc=0.80, accuracy=0.80),
        }
        table = format_results_table(results)
        assert "ModelA" in table
        assert "ModelB" in table
        # Should be sorted by F1 descending
        lines = table.strip().split("\n")
        assert lines[2].startswith("ModelA")  # First model after header

    def test_empty_results(self):
        """Test formatting with no results."""
        with pytest.raises(ValueError):
            format_results_table({})
