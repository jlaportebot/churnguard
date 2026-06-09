"""Integration tests for the full ChurnGuard pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnguard.data import DataLoader, generate_sample_data
from churnguard.evaluation import ModelEvaluator, format_results_table
from churnguard.features import FeatureEngineer
from churnguard.models import ModelRegistry


class TestFullPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def data_path(self, tmp_path: Path) -> Path:
        """Create a sample dataset."""
        df = generate_sample_data(n_rows=300, churn_rate=0.25, random_state=42)
        path = tmp_path / "churn_data.csv"
        df.to_csv(path, index=False)
        return path

    def test_full_pipeline(self, data_path: Path, tmp_path: Path):
        """Test the complete pipeline: load → features → models → evaluate."""
        # Load data
        loader = DataLoader(str(data_path), target_column="churn", random_state=42)
        X_train, X_test, y_train, y_test = loader.split()

        # Feature engineering
        engineer = FeatureEngineer(generate_interactions=False)
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)

        # Model training
        registry = ModelRegistry(models=["logistic"], random_state=42)
        results = registry.compare_all(
            X_train_tf,
            X_test_tf,
            y_train,
            y_test,
            feature_names=list(X_train_tf.columns),
        )

        # Evaluation
        output_dir = tmp_path / "results"
        evaluator = ModelEvaluator(output_dir=output_dir, save_plots=False, save_json=True)
        for _name, result in results.items():
            evaluator.evaluate_model(result, y_test)

        # Assertions
        assert len(results) == 1
        best = registry.get_best(results, metric="f1")
        assert best.f1 > 0
        assert best.roc_auc > 0.5

        # Check JSON output
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_multi_model_comparison(self, data_path: Path, tmp_path: Path):
        """Test comparing multiple models through the full pipeline."""
        from churnguard.models import GradientBoostingChurnModel, RandomForestChurnModel

        loader = DataLoader(str(data_path), target_column="churn")
        X_train, X_test, y_train, y_test = loader.split()

        engineer = FeatureEngineer(generate_interactions=False)
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)

        registry = ModelRegistry(random_state=42)
        # Use small models for speed
        registry._model_instances["random_forest"] = RandomForestChurnModel(
            n_estimators=10, max_depth=5
        )
        registry._model_instances["gradient_boosting"] = GradientBoostingChurnModel(
            n_estimators=10, max_depth=3
        )

        results = registry.compare_all(
            X_train_tf,
            X_test_tf,
            y_train,
            y_test,
            feature_names=list(X_train_tf.columns),
        )

        assert len(results) == 3

        # Format and check results table
        table = format_results_table(results)
        assert "Logistic" in table
        assert "Random Forest" in table
        assert "Gradient Boosting" in table

        # Comparison table
        comp = registry.comparison_table(results)
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) == 3

    def test_pipeline_with_config(self, data_path: Path, tmp_path: Path):
        """Test pipeline with custom configuration."""
        from churnguard.utils import get_config, merge_configs

        config = get_config()
        custom = merge_configs(
            config,
            {
                "features": {"scaling": "minmax", "generate_interactions": False},
            },
        )

        loader = DataLoader(str(data_path), target_column="churn")
        X_train, X_test, y_train, y_test = loader.split()

        feat_config = custom["features"]
        engineer = FeatureEngineer(
            scaling=feat_config["scaling"],
            generate_interactions=feat_config["generate_interactions"],
        )
        X_train_tf = engineer.fit_transform(X_train)
        engineer.transform(X_test)

        assert X_train_tf.shape[0] == len(X_train)

    def test_feature_selection_pipeline(self, data_path: Path):
        """Test pipeline with feature selection."""
        from churnguard.features import FeatureSelector

        loader = DataLoader(str(data_path), target_column="churn")
        X_train, X_test, y_train, y_test = loader.split()

        engineer = FeatureEngineer(generate_interactions=False)
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)

        # Select features
        selector = FeatureSelector(method="variance", threshold=0.01)
        X_train_sel = selector.fit_transform(X_train_tf, y_train)
        X_test_sel = selector.transform(X_test_tf)

        assert X_train_sel.shape[1] <= X_train_tf.shape[1]

        # Train model on selected features
        registry = ModelRegistry(models=["logistic"])
        result = registry.train_and_evaluate("logistic", X_train_sel, X_test_sel, y_train, y_test)
        assert result.f1 > 0

    def test_sample_data_pipeline(self, tmp_path: Path):
        """Test pipeline starting from sample data generation."""
        # Generate
        df = generate_sample_data(n_rows=200, churn_rate=0.3, random_state=42)
        path = tmp_path / "generated.csv"
        df.to_csv(path, index=False)

        # Load
        loader = DataLoader(str(path), target_column="churn")
        X_train, X_test, y_train, y_test = loader.split()

        # Features
        engineer = FeatureEngineer(generate_interactions=False)
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)

        # Model
        registry = ModelRegistry(models=["logistic"])
        result = registry.train_and_evaluate("logistic", X_train_tf, X_test_tf, y_train, y_test)

        assert result.roc_auc > 0.5  # Should be better than random
