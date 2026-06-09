"""ChurnGuard CLI — Command-line interface for churn prediction."""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from churnguard import __version__
from churnguard.data import DataLoader, generate_sample_data
from churnguard.evaluation import ModelEvaluator, format_results_table
from churnguard.features import FeatureEngineer
from churnguard.models import ModelRegistry
from churnguard.utils import ensure_dir, get_config, setup_logging

console = Console()
logger = logging.getLogger(__name__)


def _version_option(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"ChurnGuard v{__version__}")
        sys.exit(0)


@click.group()
@click.option(
    "--version", is_flag=True, callback=_version_option, is_eager=True, help="Show version."
)
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
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Models to train (logistic, random_forest, gradient_boosting).",
)
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

    console.print(
        Panel(f"[bold blue]ChurnGuard v{__version__}[/bold blue]", title="Churn Prediction")
    )

    # Load data
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Loading data...", total=None)
        loader = DataLoader(data_path, target_column=target, random_state=seed)
        X_train, X_test, y_train, y_test = loader.split()
        progress.update(
            task,
            description=f"Loaded {len(X_train) + len(X_test)} rows, target: '{loader.target_name}'",
        )

    console.print(
        f"  Dataset: [green]{len(X_train) + len(X_test)}[/green] rows, [green]{len(X_train.columns)}[/green] features"
    )
    console.print(
        f"  Target:  [green]{loader.target_name}[/green] (churn rate: {y_train.mean():.1%})"
    )
    console.print(f"  Split:   {len(X_train)} train / {len(X_test)} test")

    # Feature engineering
    feat_config = config.get("features", {})
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Engineering features...", total=None)
        engineer = FeatureEngineer(
            numeric_impute_strategy=feat_config.get("numeric_impute_strategy", "median"),
            scaling=feat_config.get("scaling", "standard"),
            categorical_max_cardinality=feat_config.get("categorical_max_cardinality", 20),
            generate_interactions=feat_config.get("generate_interactions", True),
        )
        X_train_tf = engineer.fit_transform(X_train)
        X_test_tf = engineer.transform(X_test)
        progress.update(
            task, description=f"Features: {X_train_tf.shape[1]} columns after engineering"
        )

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
                progress.update(
                    task, description=f"[green]✓[/green] {model_name}: F1={result.f1:.4f}"
                )
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
    best = registry.get_best(
        results, metric=config.get("evaluation", {}).get("primary_metric", "f1")
    )
    console.print(
        f"\n  [bold green]Best model: {best.model_name}[/bold green] (F1={best.f1:.4f}, AUC={best.roc_auc:.4f})"
    )

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
    results = registry.compare_all(
        X_train_tf, X_test_tf, y_train, y_test, feature_names=list(X_train_tf.columns)
    )

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
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to saved model (.joblib).",
)
@click.option("--output", "-o", type=click.Path(), help="Output CSV path for predictions.")
@click.option(
    "--threshold", type=float, default=0.5, help="Probability threshold for churn classification."
)
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
    console.print(
        f"  Predicted [red]{n_churn}[/red] churners out of [green]{len(df)}[/green] customers ({n_churn / len(df):.1%})"
    )

    if output:
        result_df.to_csv(output, index=False)
        console.print(f"  Saved to [blue]{output}[/blue]")
    else:
        console.print(
            result_df[
                ["churn_prediction"] + (["churn_probability"] if proba is not None else [])
            ].to_string()
        )


@main.command()
@click.argument("reference_path", type=click.Path(exists=True))
@click.argument("current_path", type=click.Path(exists=True))
@click.option("--target", "-t", default="churn", help="Target column name.")
@click.option("--output", "-o", type=click.Path(), help="Output directory for monitoring report.")
@click.option("--psi-threshold", type=float, default=0.10, help="PSI threshold for drift detection.")
@click.option("--ks-alpha", type=float, default=0.05, help="Alpha level for KS test.")
@click.option("--report/--no-report", default=True, help="Generate HTML report.")
@click.option("--concept-drift/--no-concept-drift", default=False, help="Run concept drift detection (requires --model).")
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to saved model for concept drift detection.")
@click.option("--alert-config", type=click.Path(), help="Path to alert configuration JSON.")
@click.pass_context
def monitor(
    ctx: click.Context,
    reference_path: str,
    current_path: str,
    target: str,
    output: Optional[str],
    psi_threshold: float,
    ks_alpha: float,
    report: bool,
    concept_drift: bool,
    model: Optional[str],
    alert_config: Optional[str],
) -> None:
    """Monitor model performance and detect data drift.

    Compare a REFERENCE dataset (training data) against a CURRENT dataset
    (production data) to detect distribution shifts and performance degradation.

    \b
    Example:
      churnguard monitor reference.csv current.csv --output monitoring_output/
    """
    from churnguard.monitoring import (
        DataDriftDetector,
        PerformanceMonitor,
        AlertManager,
        MonitoringReport,
        MonitoringReportConfig,
        ADWIN,
        DDM,
        EDDM,
        DEFAULT_CHURN_ALERT_RULES,
    )

    output_dir = ensure_dir(output) if output else ensure_dir("./churnguard_monitoring")

    console.print(Panel(f"[bold blue]ChurnGuard Monitoring v{__version__}[/bold blue]", title="Model Monitoring"))

    # Load data
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading datasets...", total=None)
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        progress.update(task, description=f"Reference: {len(reference_df)} rows | Current: {len(current_df)} rows")

    # --- Data drift detection ---
    console.print("\n[bold]📊 Data Drift Detection[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Running drift tests...", total=None)
        detector = DataDriftDetector(
            psi_threshold=psi_threshold,
            ks_alpha=ks_alpha,
        )
        drift_result = detector.detect(reference_df, current_df)
        progress.update(task, description=f"Drift check complete: {drift_result.n_features_drifted}/{drift_result.n_features_tested} features drifted")

    # Display drift summary
    severity_color = {
        "none": "green",
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    color = severity_color.get(drift_result.overall_severity.value, "white")
    console.print(f"  Overall severity: [{color}]{drift_result.overall_severity.value.upper()}[/{color}]")
    console.print(f"  Drift score: {drift_result.drift_score:.4f}")
    console.print(f"  Features drifted: {drift_result.n_features_drifted}/{drift_result.n_features_tested}")

    if drift_result.drifted_features():
        table = Table(title="Drifted Features")
        table.add_column("Feature", style="bold")
        table.add_column("PSI", justify="right")
        table.add_column("KS p-value", justify="right")
        table.add_column("Severity", justify="center")

        for psi_r in drift_result.psi_results:
            if psi_r.severity.value != "none":
                ks_p = next(
                    (r.p_value for r in drift_result.ks_results if r.feature_name == psi_r.feature_name),
                    None,
                )
                table.add_row(
                    psi_r.feature_name,
                    f"{psi_r.psi_value:.4f}",
                    f"{ks_p:.4f}" if ks_p is not None else "N/A",
                    psi_r.severity.value.upper(),
                )

        console.print(table)

    # --- Performance monitoring (if target column exists) ---
    perf_monitor = None
    if target in current_df.columns and target in reference_df.columns:
        console.print("\n[bold]📈 Performance Monitoring[/bold]")
        import joblib

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Evaluating performance...", total=None)

            perf_monitor = PerformanceMonitor(model_name="churn_model")

            # Split reference for baseline evaluation
            from sklearn.model_selection import train_test_split
            ref_X = reference_df.drop(columns=[target])
            ref_y = reference_df[target]
            cur_X = current_df.drop(columns=[target])
            cur_y = current_df[target]

            if model:
                model_obj = joblib.load(model)
                if hasattr(model_obj, "predict_proba"):
                    ref_proba = model_obj.predict_proba(ref_X)[:, 1]
                    cur_proba = model_obj.predict_proba(cur_X)[:, 1]
                else:
                    ref_proba = None
                    cur_proba = None

                ref_pred = model_obj.predict(ref_X)
                cur_pred = model_obj.predict(cur_X)
            else:
                # Simple heuristic baseline: use reference mean as threshold
                ref_mean = ref_y.mean()
                ref_pred = (ref_X.mean(axis=1) > ref_mean).astype(int) if len(ref_X.columns) > 0 else np.zeros(len(ref_y), dtype=int)
                cur_pred = (cur_X.mean(axis=1) > ref_mean).astype(int) if len(cur_X.columns) > 0 else np.zeros(len(cur_y), dtype=int)
                ref_proba = None
                cur_proba = None

            # Baseline snapshot
            baseline_snapshot = perf_monitor.evaluate(
                y_true=ref_y.values,
                y_pred=ref_pred,
                y_proba=ref_proba,
                timestamp="baseline",
            )

            # Current snapshot
            current_snapshot = perf_monitor.evaluate(
                y_true=cur_y.values,
                y_pred=cur_pred,
                y_proba=cur_proba,
            )

            progress.update(task, description=f"Baseline F1={baseline_snapshot.f1:.4f} | Current F1={current_snapshot.f1:.4f}")

        console.print(f"  Baseline F1: [green]{baseline_snapshot.f1:.4f}[/green]")
        console.print(f"  Current  F1: [{'red' if current_snapshot.f1 < baseline_snapshot.f1 else 'green'}]{current_snapshot.f1:.4f}[/{'red' if current_snapshot.f1 < baseline_snapshot.f1 else 'green'}]")

        if perf_monitor.alerts:
            console.print(f"\n  [yellow]⚠ {len(perf_monitor.alerts)} performance alert(s)[/yellow]")
            for alert in perf_monitor.alerts:
                console.print(f"    • {alert.summary()}")

    # --- Concept drift detection (optional) ---
    concept_drift_results = None
    if concept_drift and model and target in current_df.columns:
        console.print("\n[bold]🔄 Concept Drift Detection[/bold]")
        import joblib

        model_obj = joblib.load(model)
        cur_X = current_df.drop(columns=[target])
        cur_y = current_df[target]
        predictions = model_obj.predict(cur_X)

        adwin = ADWIN()
        ddm = DDM()
        eddm = EDDM()

        adwin_results = adwin.update_batch(predictions, cur_y.values)
        ddm_results = ddm.update_batch(predictions, cur_y.values)
        eddm_results = eddm.update_batch(predictions, cur_y.values)

        concept_drift_results = {
            "ADWIN": adwin.detect(),
            "DDM": ddm.detect(),
            "EDDM": eddm.detect(),
        }

        for name, result in concept_drift_results.items():
            console.print(f"  {result.summary()}")

    # --- Alert management ---
    alert_manager = AlertManager(rules=DEFAULT_CHURN_ALERT_RULES)
    metrics_for_alerts = {"drift_score": drift_result.drift_score}

    if perf_monitor and perf_monitor.snapshots:
        latest = perf_monitor.snapshots[-1]
        metrics_for_alerts.update({
            "f1": latest.f1,
            "roc_auc": latest.roc_auc,
            "churn_rate": latest.churn_rate,
        })

    new_alerts = alert_manager.check(metrics_for_alerts)

    if new_alerts:
        console.print(f"\n[bold yellow]🔔 {len(new_alerts)} new alert(s)[/bold yellow]")
        for alert in new_alerts:
            console.print(f"  • {alert.summary()}")
    else:
        console.print(f"\n[green]✓ No new alerts[/green]")

    # --- HTML report ---
    if report:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Generating HTML report...", total=None)
            report_gen = MonitoringReport(MonitoringReportConfig(
                title=f"ChurnGuard Monitoring Report — {reference_path}",
                include_concept_drift=concept_drift_results is not None,
            ))
            report_path = output_dir / "monitoring_report.html"
            report_gen.generate(
                drift_result=drift_result,
                performance_monitor=perf_monitor,
                alert_manager=alert_manager,
                concept_drift_results=concept_drift_results,
                output_path=report_path,
            )
            progress.update(task, description=f"Report saved to {report_path}")

    # Save JSON results
    import json as json_mod
    results_json = {
        "drift": drift_result.to_dict(),
        "alerts": [a.to_dict() for a in new_alerts],
    }
    json_path = output_dir / "monitoring_results.json"
    json_path.write_text(json_mod.dumps(results_json, indent=2, default=str))
    console.print(f"\n Results saved to [blue]{output_dir}[/blue]")


if __name__ == "__main__":
    main()


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", help="Target column name.")
@click.option("--output", "-o", type=click.Path(), help="Output directory.")
@click.option("--models", "-m", multiple=True, help="Models to train.")
@click.option("--optimize-threshold", is_flag=True, help="Optimize decision threshold.")
@click.option(
    "--cost-ratio",
    type=float,
    default=5.0,
    help="Cost of FN / cost of FP for threshold optimization.",
)
@click.option("--explain", is_flag=True, help="Generate SHAP explanations.")
@click.option(
    "--revenue", type=float, default=100.0, help="Revenue per customer for business impact."
)
@click.option("--intervention-cost", type=float, default=10.0, help="Cost per intervention.")
@click.option("--success-rate", type=float, default=0.3, help="Intervention success rate.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def pipeline(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    output: Optional[str],
    models: tuple[str, ...],
    optimize_threshold: bool,
    cost_ratio: float,
    explain: bool,
    revenue: float,
    intervention_cost: float,
    success_rate: float,
    seed: int,
) -> None:
    """Run the full churn prediction pipeline.

    End-to-end workflow: data → features → training → evaluation →
    threshold optimization → explainability → business impact → report.
    """
    from churnguard.pipeline import ChurnPipeline, PipelineConfig
    from churnguard.threshold import CostMatrix

    output_dir = ensure_dir(output) if output else ensure_dir("./churnguard_output")

    config = PipelineConfig(
        target=target or "churn",
        models=list(models) if models else ["logistic", "random_forest", "gradient_boosting"],
        optimize_threshold=optimize_threshold,
        cost_matrix=CostMatrix(cost_fn=cost_ratio, cost_fp=1.0) if optimize_threshold else None,
        explain=explain,
        revenue_per_customer=revenue,
        intervention_cost=intervention_cost,
        intervention_success_rate=success_rate,
        output_dir=str(output_dir),
        random_state=seed,
    )

    console.print(
        Panel(
            f"[bold blue]ChurnGuard Pipeline v{__version__}[/bold blue]",
            title="End-to-End Churn Prediction",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)
        pipe = ChurnPipeline(config=config)
        result = pipe.run(data_path, target=target or "churn")
        progress.update(task, description="[green]Pipeline complete[/green]")

    # Display best model
    console.print(
        f"\n  [bold green]Best model:[/bold green] {result.best_model_name} (F1={result.best_result.f1:.4f})"
    )

    # Display model comparison
    table = Table(title="Model Results")
    table.add_column("Model", style="bold")
    table.add_column("F1", justify="right", style="green")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    for name, r in sorted(result.model_results.items(), key=lambda x: x[1].f1, reverse=True):
        table.add_row(
            r.model_name, f"{r.f1:.4f}", f"{r.roc_auc:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}"
        )
    console.print(table)

    # Threshold optimization results
    if result.threshold_result:
        tr = result.threshold_result
        console.print(
            f"\n  [bold yellow]Optimal threshold:[/bold yellow] {tr.optimal_threshold:.4f} (strategy: {tr.strategy})"
        )
        console.print(f"  F1 at optimal: {tr.f1_at_threshold:.4f}")

    # Business impact
    if "business_impact" in result.run_info:
        bi = result.run_info["business_impact"]
        console.print("\n  [bold cyan]Business Impact:[/bold cyan]")
        console.print(f"  Revenue saved: ${bi['revenue_saved']:,.0f}")
        console.print(f"  Intervention cost: ${bi['intervention_cost']:,.0f}")
        console.print(f"  Net value: ${bi['net_value']:,.0f}")
        console.print(f"  ROI: {bi['roi_percent']:.1f}%")

    # SHAP explanation summary
    if result.global_explanation:
        top5 = list(result.global_explanation.feature_importance.items())[:5]
        console.print("\n  [bold magenta]Top 5 Features (SHAP):[/bold magenta]")
        for feat, imp in top5:
            console.print(f"  {feat}: {imp:.4f}")

    console.print(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")
    console.print(f"  Report saved to [blue]{output_dir}[/blue]")


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", help="Target column name.")
@click.option(
    "--model", "-m", default="logistic", help="Model to train for threshold optimization."
)
@click.option(
    "--strategy",
    type=click.Choice(["f1", "youden", "cost_sensitive"]),
    default="f1",
    help="Optimization strategy.",
)
@click.option(
    "--cost-ratio", type=float, default=5.0, help="Cost FN / Cost FP (for cost_sensitive strategy)."
)
@click.option("--output", "-o", type=click.Path(), help="Output directory for threshold report.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def threshold(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    model: str,
    strategy: str,
    cost_ratio: float,
    output: Optional[str],
    seed: int,
) -> None:
    """Find the optimal decision threshold for churn classification.

    Trains a model and evaluates multiple threshold optimization strategies.
    """
    from churnguard.threshold import CostMatrix, ThresholdOptimizer

    loader = DataLoader(data_path, target_column=target, random_state=seed)
    X_train, X_test, y_train, y_test = loader.split()

    engineer = FeatureEngineer()
    X_train_tf = engineer.fit_transform(X_train)
    X_test_tf = engineer.transform(X_test)

    registry = ModelRegistry(models=[model], random_state=seed)
    train_result = registry.train_and_evaluate(model, X_train_tf, X_test_tf, y_train, y_test)

    console.print(f"\n  Model: [green]{train_result.model_name}[/green] (F1={train_result.f1:.4f})")

    # Optimize threshold
    optimizer = ThresholdOptimizer()
    y_proba = train_result.y_proba

    strategies = ["f1", "youden", "cost_sensitive"] if strategy == "f1" else [strategy]

    table = Table(title="Threshold Optimization Results")
    table.add_column("Strategy", style="bold")
    table.add_column("Threshold", justify="right")
    table.add_column("F1", justify="right", style="green")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")

    for strat in strategies:
        if strat == "cost_sensitive":
            result = optimizer.optimize(
                y_test,
                y_proba,
                strategy=strat,
                cost_matrix=CostMatrix(cost_fn=cost_ratio, cost_fp=1.0),
            )
        else:
            result = optimizer.optimize(y_test, y_proba, strategy=strat)
        table.add_row(
            strat,
            f"{result.optimal_threshold:.4f}",
            f"{result.f1_at_threshold:.4f}",
            f"{result.precision_at_threshold:.4f}",
            f"{result.recall_at_threshold:.4f}",
        )

    console.print(table)

    if output:
        output_dir = ensure_dir(output)
        report = {"model": model, "strategies": {}}
        for strat in strategies:
            if strat == "cost_sensitive":
                result = optimizer.optimize(
                    y_test,
                    y_proba,
                    strategy=strat,
                    cost_matrix=CostMatrix(cost_fn=cost_ratio, cost_fp=1.0),
                )
            else:
                result = optimizer.optimize(y_test, y_proba, strategy=strat)
            report["strategies"][strat] = {
                "optimal_threshold": result.optimal_threshold,
                "f1": result.f1_at_threshold,
                "precision": result.precision_at_threshold,
                "recall": result.recall_at_threshold,
            }
        (output_dir / "threshold_report.json").write_text(json.dumps(report, indent=2))
        console.print(f"\n  Report saved to [blue]{output_dir / 'threshold_report.json'}[/blue]")


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", help="Target column name.")
@click.option("--model", "-m", default="logistic", help="Model to explain.")
@click.option("--customer-id", type=int, default=0, help="Index of customer to explain in detail.")
@click.option("--top-n", type=int, default=10, help="Number of top features to display.")
@click.option("--output", "-o", type=click.Path(), help="Output directory for explanation report.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def explain(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    model: str,
    customer_id: int,
    top_n: int,
    output: Optional[str],
    seed: int,
) -> None:
    """Explain model predictions using SHAP values.

    Shows global feature importance and per-customer explanations.
    """
    from churnguard.explainability import ChurnExplainer

    loader = DataLoader(data_path, target_column=target, random_state=seed)
    X_train, X_test, y_train, y_test = loader.split()

    engineer = FeatureEngineer()
    X_train_tf = engineer.fit_transform(X_train)
    X_test_tf = engineer.transform(X_test)

    registry = ModelRegistry(models=[model], random_state=seed)
    registry.train_and_evaluate(model, X_train_tf, X_test_tf, y_train, y_test)

    # Get the trained sklearn model
    churn_model = registry.get_model(model)
    inner_model = churn_model._model if hasattr(churn_model, "_model") else churn_model

    console.print(f"\n  Computing SHAP explanations for [green]{churn_model.name}[/green]...")

    explainer = ChurnExplainer(model=inner_model)
    explainer.fit(X_train_tf, feature_names=list(X_train_tf.columns))

    # Global explanation
    global_exp = explainer.explain_global(
        X_test_tf.iloc[:100] if len(X_test_tf) > 100 else X_test_tf
    )

    table = Table(title=f"Top {top_n} Features (SHAP)")
    table.add_column("Feature", style="bold")
    table.add_column("Importance", justify="right", style="green")
    for feat, imp in list(global_exp.feature_importance.items())[:top_n]:
        table.add_row(feat, f"{imp:.4f}")
    console.print(table)

    # Customer-level explanation
    if customer_id < len(X_test_tf):
        cust_exp = explainer.explain_customer(X_test_tf, customer_index=customer_id)
        console.print(f"\n  [bold]Customer #{customer_id} explanation:[/bold]")
        console.print(f"  Churn probability contribution: {cust_exp.base_value:.4f}")
        cust_table = Table(title=f"Top features for Customer #{customer_id}")
        cust_table.add_column("Feature", style="bold")
        cust_table.add_column("SHAP value", justify="right")
        cust_table.add_column("Direction", justify="right")
        top_features = sorted(
            cust_exp.feature_contribution.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]
        for feat, val in top_features:
            direction = "[red]↑ churn[/red]" if val > 0 else "[green]↓ churn[/green]"
            cust_table.add_row(feat, f"{val:+.4f}", direction)
        console.print(cust_table)

    if output:
        output_dir = ensure_dir(output)
        report = {
            "model": churn_model.name,
            "global_importance": dict(list(global_exp.feature_importance.items())[:top_n]),
        }
        if customer_id < len(X_test_tf):
            report["customer_explanation"] = {
                "customer_index": customer_id,
                "base_value": cust_exp.base_value,
                "top_features": {k: v for k, v in top_features},
            }
        (output_dir / "explanation_report.json").write_text(json.dumps(report, indent=2))
        console.print(f"\n  Report saved to [blue]{output_dir / 'explanation_report.json'}[/blue]")
