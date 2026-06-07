"""Integration tests for the end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churnguard.data import generate_sample_data
from churnguard.pipeline import ChurnPipeline, PipelineConfig, run_pipeline
from churnguard.threshold import CostMatrix


@pytest.fixture
def sample_csv(tmp_path):
    """Generate a sample CSV file."""
    df = generate_sample_data(n_rows=300, churn_rate=0.25, random_state=42)
    csv_path = tmp_path / "churn_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestPipelineIntegration:
    """Full end-to-end pipeline integration tests."""

    def test_full_pipeline_logistic(self, sample_csv, tmp_path):
        """Run full pipeline with logistic regression."""
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(tmp_path / "output"),
        )
        assert result.best_model_name == "Logistic Regression"
        assert result.model_results["logistic"].f1 > 0
        assert result.model_results["logistic"].roc_auc > 0.5

    def test_full_pipeline_with_threshold(self, sample_csv, tmp_path):
        """Run pipeline with threshold optimization."""
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(tmp_path / "output"),
            optimize_threshold=True,
            threshold_method="youden",
        )
        assert result.threshold_result is not None
        assert result.threshold_result.method == "youden"
        assert 0 < result.threshold_result.threshold < 1

    def test_full_pipeline_all_models(self, sample_csv, tmp_path):
        """Run pipeline with all three built-in models."""
        result = run_pipeline(
            sample_csv,
            target="churn",
            model=["logistic", "random_forest", "gradient_boosting"],
            output_dir=str(tmp_path / "output"),
        )
        assert len(result.model_results) == 3
        # Best model should be one of the three
        assert result.best_model_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]

    def test_pipeline_business_impact(self, sample_csv, tmp_path):
        """Verify business impact metrics are computed."""
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(tmp_path / "output"),
        )
        assert "business_impact" in result.run_info
        bi = result.run_info["business_impact"]
        assert "revenue_saved" in bi
        assert "intervention_cost" in bi
        assert "net_value" in bi

    def test_pipeline_report_json(self, sample_csv, tmp_path):
        """Verify JSON report is properly structured."""
        out_dir = tmp_path / "output"
        result = run_pipeline(
            sample_csv,
            target="churn",
            model="logistic",
            output_dir=str(out_dir),
        )
        report_path = out_dir / "pipeline_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert "best_model" in report
        assert "models" in report
        assert "run_info" in report
        assert "logistic" in report["models"]

    def test_pipeline_with_cost_threshold(self, sample_csv, tmp_path):
        """Test pipeline with cost-based threshold optimization."""
        config = PipelineConfig(
            model="logistic",
            optimize_threshold=True,
            threshold_method="cost",
            cost_matrix=CostMatrix(tp_benefit=200, fp_cost=20, fn_cost=200),
            output_dir=str(tmp_path / "output"),
            save_plots=False,
        )
        pipeline = ChurnPipeline(config=config)
        result = pipeline.run(sample_csv, target="churn")
        assert result.threshold_result is not None
        assert result.threshold_result.method == "cost"
        assert "expected_value" in result.threshold_result.metrics
