"""Tests for the pipeline module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from churnguard.data import generate_sample_data
from churnguard.pipeline import (
    ChurnPipeline,
    PipelineConfig,
    PipelineResult,
    compute_business_impact,
    run_pipeline,
)
from churnguard.threshold import CostMatrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path):
    """Generate a sample CSV file for pipeline testing."""
    df = generate_sample_data(n_rows=200, churn_rate=0.25, random_state=42)
    csv_path = tmp_path / "test_churn.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_data():
    """Generate sample data as DataFrame."""
    return generate_sample_data(n_rows=200, churn_rate=0.25, random_state=42)


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.model == "logistic"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.optimize_threshold is False
        assert config.compute_explanations is False

    def test_custom_config(self):
        config = PipelineConfig(
            model=["logistic", "random_forest"],
            test_size=0.3,
            optimize_threshold=True,
            threshold_method="youden",
        )
        assert config.get_model_names() == ["logistic", "random_forest"]
        assert config.test_size == 0.3

    def test_single_model_as_list(self):
        config = PipelineConfig(model="gradient_boosting")
        assert config.get_model_names() == ["gradient_boosting"]

    def test_model_list(self):
        config = PipelineConfig(model=["logistic", "random_forest", "gradient_boosting"])
        names = config.get_model_names()
        assert len(names) == 3


# ---------------------------------------------------------------------------
# compute_business_impact tests
# ---------------------------------------------------------------------------


class TestComputeBusinessImpact:
    def test_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        impact = compute_business_impact(
            y_true,
            y_proba,
            threshold=0.5,
            revenue_per_customer=100,
            intervention_cost=10,
            intervention_success_rate=0.3,
        )
        assert impact["true_positives"] == 3
        assert impact["false_positives"] == 0
        assert impact["false_negatives"] == 0
        assert impact["true_negatives"] == 3
        assert impact["revenue_saved"] == pytest.approx(3 * 0.3 * 100, abs=0.01)
        assert impact["intervention_cost"] == 3 * 10

    def test_all_flagged(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])
        impact = compute_business_impact(
            y_true,
            y_proba,
            threshold=0.5,
            revenue_per_customer=100,
            intervention_cost=10,
        )
        assert impact["n_flagged"] == 4
        assert impact["false_positives"] == 2

    def test_none_flagged(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        impact = compute_business_impact(
            y_true,
            y_proba,
            threshold=0.8,
        )
        assert impact["n_flagged"] == 0
        assert impact["false_negatives"] == 2

    def test_roi_calculation(self):
        y_true = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.6, 0.7, 0.8])
        impact = compute_business_impact(y_true, y_proba, threshold=0.5)
        assert "roi_percent" in impact
        assert "value_vs_baseline_percent" in impact


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_summary_with_no_results(self):
        result = PipelineResult()
        s = result.summary()
        assert isinstance(s, str)

    def test_save_report(self, tmp_path):
        from churnguard.models.base import ModelResult

        result = PipelineResult(
            model_results={
                "logistic": ModelResult(model_name="logistic", f1=0.8, roc_auc=0.9),
            },
            best_model_name="logistic",
            run_info={"timestamp": "2026-01-01", "n_rows": 100},
            elapsed_seconds=1.5,
        )
        report_path = tmp_path / "report.json"
        result.save_report(report_path)
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["best_model"] == "logistic"
        assert data["elapsed_seconds"] == 1.5


# ---------------------------------------------------------------------------
# ChurnPipeline integration tests
# ---------------------------------------------------------------------------


class TestChurnPipeline:
    def test_run_with_logistic(self, sample_csv, tmp_path):
        config = PipelineConfig(
            model="logistic",
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")

        assert result.best_model_name == "Logistic Regression"
        assert "logistic" in result.model_results
        assert result.model_results["logistic"].f1 > 0
        assert result.elapsed_seconds > 0
        assert result.run_info["n_rows"] > 0

    def test_run_with_multiple_models(self, sample_csv, tmp_path):
        config = PipelineConfig(
            model=["logistic", "random_forest"],
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")

        assert len(result.model_results) == 2
        assert result.best_model_name in ["Logistic Regression", "Random Forest"]

    def test_run_with_threshold_optimization(self, sample_csv, tmp_path):
        config = PipelineConfig(
            model="logistic",
            optimize_threshold=True,
            threshold_method="f1",
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")

        assert result.threshold_result is not None
        assert result.threshold_result.method == "f1"
        assert 0 < result.threshold_result.threshold < 1

    def test_run_with_cost_threshold(self, sample_csv, tmp_path):
        config = PipelineConfig(
            model="logistic",
            optimize_threshold=True,
            threshold_method="cost",
            cost_matrix=CostMatrix(tp_benefit=200, fp_cost=15, fn_cost=150),
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")
        assert result.threshold_result is not None

    def test_pipeline_saves_report(self, sample_csv, tmp_path):
        out_dir = tmp_path / "output"
        config = PipelineConfig(
            model="logistic",
            output_dir=str(out_dir),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")

        report_path = out_dir / "pipeline_report.json"
        assert report_path.exists()

    def test_pipeline_saves_comparison(self, sample_csv, tmp_path):
        out_dir = tmp_path / "output"
        config = PipelineConfig(
            model="logistic",
            output_dir=str(out_dir),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")

        comparison_path = out_dir / "model_comparison.csv"
        assert comparison_path.exists()

    def test_predict_after_run(self, sample_csv, tmp_path):
        config = PipelineConfig(
            model="logistic",
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        pipeline.run(sample_csv, target="churn")

        # Create a new CSV for prediction
        df = generate_sample_data(n_rows=50, random_state=99)
        pred_path = tmp_path / "new_customers.csv"
        df.to_csv(pred_path, index=False)

        predictions = pipeline.predict(str(pred_path))
        assert "churn_probability" in predictions.columns
        assert "churn_label" in predictions.columns
        assert len(predictions) == 50

    def test_predict_before_run_raises(self):
        pipeline = ChurnPipeline()
        with pytest.raises(RuntimeError, match="Pipeline must be run"):
            pipeline.predict("nonexistent.csv")


# ---------------------------------------------------------------------------
# run_pipeline convenience function tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_convenience_function(self, sample_csv, tmp_path):
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(tmp_path / "output"),
        )
        assert isinstance(result, PipelineResult)
        assert result.best_model_name is not None

    def test_with_all_options(self, sample_csv, tmp_path):
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(tmp_path / "output"),
            optimize_threshold=True,
            threshold_method="youden",
            random_state=42,
        )
        assert result.threshold_result is not None
