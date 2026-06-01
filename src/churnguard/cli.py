"""ChurnGuard CLI — Command-line interface for churn prediction."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from churnguard import __version__
from churnguard.data import DataLoader, generate_sample_data
from churnguard.features import FeatureEngineer
from churnguard.models import ModelRegistry
from churnguard.evaluation import ModelEvaluator, format_results_table
from churnguard.utils import setup_logging, get_config, ensure_dir

console = Console()
logger = logging.getLogger(__name__)


def _version_option(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"ChurnGuard v{__version__}")
        sys.exit(0)


@click.group()
@click.option("--version", is_flag=True, callback=_version_option, is_eager=True, help="Show version.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--config", "-c", type=click.Path(), help="Path to YAML config file.")
@click.pass_context
def main(ctx: click.Context, version: bool, verbose: bool, config: Optional[str]) -> None:
    """ChurnGuard — Customer churn prediction CLI.

    Analyze customer data, train ML models, and predict churn risk.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config(config)
    ctx.obj["verbose"] = verbose
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", help="Target column name (auto-detected if not specified).")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results.")
@click.option("--models", "-m", multiple=True, help="Models to train (logistic, random_forest, gradient_boosting).")
@click.option("--tune", is_flag=True, help="Enable hyperparameter tuning.")
@click.option("--no-plots", is_flag=True, help="Skip plot generation.")
@click.option("--cv-folds", type=int, default=5, help="Number of CV folds.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def analyze(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    output: Optional[str],
    models: tuple[str, ...],
    tune: bool,
    no_plots: bool,
    cv_folds: int,
    seed: int,
) -> None:
    """Analyze a dataset for churn prediction.

    Load data, engineer features, train models, and output evaluation results.
    """
    config = ctx.obj["config"]
    output_dir = ensure_dir(output) if output else ensure_dir("./churnguard_output")

    console.print(Panel(f"[bold blue]ChurnGuard v{__version__}[/bold blue]", title="Churn Prediction"))

    # Load data
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading data...", total=None)
        loader = DataLoader(data_path, target_column=target, random_state=seed)
        X_train, X_test, y_train, y_test = loader.split()
        progress.update(task, description=f"Loaded {len(X_train) + len(X_test)} rows, target: '{loader.target_name}'")

    console.print(f"  Dataset: [green]{len(X_train) + len(X_test)}[/green] rows, [green]{len(X_train.columns)}[/green] features")
    console.print(f"  Target:  [green]{loader.target_name}[/green] (churn rate: {y_train.mean():.1%})")
    console.print(f"  Split:   {len(X_train)} train / {len(X_test)} test")

    # Feature engineering
    feat_config = config.get("features", {})
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Engineering features...", total=None)
        engineer = FeatureEngineer(
            numeric_impute_strategy=feat_config.get("numeric_impute_strategy", "median"),
            scaling=feat_config.get("scaling", "standard"),
            categorical_max_cardinality=feat_config.get("categorical_max_cardinality", 20),
            generate_interactions=feat_config.get("generate_interactions", True),
        )
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)
        progress.update(task, description=f"Features: {X_train_tf.shape[1]} columns after engineering")

    console.print(f"  Engineered features: [green]{X_train_tf.shape[1]}[/green]")

    # Train models
    model_names = list(models) if models else None
    registry = ModelRegistry(
        models=model_names,
        random_state=seed,
        tune_hyperparams=tune,
        cv_folds=cv_folds,
    )

    console.print(f"\n  Training {len(registry.available_models)} models...")
    results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for model_name in registry.available_models:
            task = progress.add_task(f"Training {model_name}...", total=None)
            try:
                result = registry.train_and_evaluate(
                    model_name,
                    X_train_tf,
                    X_test_tf,
                    y_train,
                    y_test,
                    feature_names=list(X_train_tf.columns),
                )
                results[model_name] = result
                progress.update(task, description=f"[green]✓[/green] {model_name}: F1={result.f1:.4f}")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {model_name}: {e}")
                logger.error("Failed to train %s: %s", model_name, e)

    if not results:
        console.print("[red]All models failed to train![/red]")
        sys.exit(1)

    # Display results
    table = format_results_table(results)
    console.print(f"\n{table}")

    # Best model
    best = registry.get_best(results, metric=config.get("evaluation", {}).get("primary_metric", "f1"))
    console.print(f"\n  [bold green]Best model: {best.model_name}[/bold green] (F1={best.f1:.4f}, AUC={best.roc_auc:.4f})")

    # Save evaluation outputs
    evaluator = ModelEvaluator(
        output_dir=output_dir,
        save_plots=not no_plots,
        save_json=True,
    )

    for name, result in results.items():
        evaluator.evaluate_model(result, y_test)

    comparison = evaluator.compare_models(results, y_test)
    comparison.to_csv(output_dir / "model_comparison.csv")

    console.print(f"\n  Results saved to [blue]{output_dir}[/blue]")


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", help="Target column name.")
@click.option("--output", "-o", type=click.Path(), help="Output directory.")
@click.option("--cv-folds", type=int, default=5, help="CV folds.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def compare(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    output: Optional[str],
    cv_folds: int,
    seed: int,
) -> None:
    """Compare all available churn models.

    Trains all models and shows a side-by-side comparison.
    """
    output_dir = ensure_dir(output) if output else ensure_dir("./churnguard_output")

    loader = DataLoader(data_path, target_column=target, random_state=seed)
    X_train, X_test, y_train, y_test = loader.split()

    engineer = FeatureEngineer()
    X_train_tf = engineer.fit_transform(X_train)
    X_test_tf = engineer.transform(X_test)

    registry = ModelRegistry(random_state=seed, cv_folds=cv_folds)
    results = registry.compare_all(X_train_tf, X_test_tf, y_train, y_test, feature_names=list(X_train_tf.columns))

    # Rich comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="green")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Time", justify="right")

    for result in sorted(results.values(), key=lambda r: r.f1, reverse=True):
        table.add_row(
            result.model_name,
            f"{result.accuracy:.3f}",
            f"{result.precision:.3f}",
            f"{result.recall:.3f}",
            f"{result.f1:.3f}",
            f"{result.roc_auc:.3f}",
            f"{result.training_time_seconds:.1f}s",
        )

    console.print(table)

    # Save comparison
    evaluator = ModelEvaluator(output_dir=output_dir)
    df = evaluator.compare_models(results, y_test)
    df.to_csv(output_dir / "model_comparison.csv")
    console.print(f"\n  Saved to [blue]{output_dir}[/blue]")


@main.command()
@click.option("--output", "-o", type=click.Path(), required=True, help="Output CSV path.")
@click.option("--rows", "-n", type=int, default=1000, help="Number of rows.")
@click.option("--churn-rate", type=float, default=0.2, help="Approximate churn rate (0-1).")
@click.option("--seed", type=int, default=42, help="Random seed.")
def sample(output: str, rows: int, churn_rate: float, seed: int) -> None:
    """Generate sample churn data for testing.

    Creates a synthetic dataset with realistic feature distributions and churn patterns.
    """
    console.print(f"Generating [green]{rows}[/green] rows with ~{churn_rate:.0%} churn rate...")
    df = generate_sample_data(n_rows=rows, churn_rate=churn_rate, random_state=seed)
    df.to_csv(output, index=False)
    console.print(f"  Saved to [blue]{output}[/blue]")
    console.print(f"  Actual churn rate: {df['churn'].mean():.1%}")
    console.print(f"  Columns: {list(df.columns)}")


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Path to saved model (.joblib).")
@click.option("--output", "-o", type=click.Path(), help="Output CSV path for predictions.")
@click.option("--threshold", type=float, default=0.5, help="Probability threshold for churn classification.")
def score(data_path: str, model: str, output: Optional[str], threshold: float) -> None:
    """Score new customer data using a trained model.

    Loads a saved model and predicts churn probability for each customer.
    """
    import joblib

    console.print(f"Loading model from [blue]{model}[/blue]...")
    model_obj = joblib.load(model)

    console.print(f"Loading data from [blue]{data_path}[/blue]...")
    df = pd.read_csv(data_path)

    console.print("Generating predictions...")
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(df)[:, 1]
        predictions = (proba >= threshold).astype(int)
    else:
        proba = None
        predictions = model_obj.predict(df)

    result_df = df.copy()
    result_df["churn_prediction"] = predictions
    if proba is not None:
        result_df["churn_probability"] = proba

    n_churn = predictions.sum()
    console.print(f"  Predicted [red]{n_churn}[/red] churners out of [green]{len(df)}[/green] customers ({n_churn/len(df):.1%})")

    if output:
        result_df.to_csv(output, index=False)
        console.print(f"  Saved to [blue]{output}[/blue]")
    else:
        console.print(result_df[["churn_prediction"] + (["churn_probability"] if proba is not None else [])].to_string())


if __name__ == "__main__":
    main()
