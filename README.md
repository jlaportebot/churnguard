# ChurnGuard

Customer churn prediction CLI and library with ML model comparison, threshold optimization, and SHAP explainability.

## Features

- **CLI tool** for quick churn analysis from the command line
- **Feature engineering pipeline** with automatic preprocessing
- **Multiple ML models**: Logistic Regression, Random Forest, Gradient Boosting
- **Model evaluation & comparison** with rich metrics output
- **Threshold optimization**: F1-optimal, Youden's J, cost-sensitive strategies
- **SHAP explainability**: global feature importance and per-customer explanations
- **Business impact analysis**: ROI, revenue saved, intervention cost modeling
- **End-to-end pipeline**: data → features → training → threshold → explain → report
- **Visualization** of feature importance, confusion matrices, and ROC curves
- **Configurable** via YAML or CLI flags
- **Extensible** model registry via entry points

## Installation

```bash
pip install churnguard
```

With visualization support:

```bash
pip install churnguard[viz]
```

## Quick Start

### CLI

```bash
# Analyze a CSV dataset
churnguard analyze data.csv --target churn --output results/

# Run the full pipeline (training + threshold + explain + business impact)
churnguard pipeline data.csv --target churn --optimize-threshold --explain

# Find optimal decision threshold
churnguard threshold data.csv --target churn --strategy cost_sensitive --cost-ratio 5.0

# Explain model predictions with SHAP
churnguard explain data.csv --target churn --customer-id 42 --top-n 10

# Compare all models
churnguard compare data.csv --target churn

# Generate sample data for testing
churnguard sample --output sample_data.csv --rows 1000

# Score new customers
churnguard score new_customers.csv --model results/best_model.joblib
```

### Python API

```python
from churnguard.pipeline import ChurnPipeline, PipelineConfig
from churnguard.threshold import ThresholdOptimizer, CostMatrix

# Run the full pipeline
config = PipelineConfig(
    target="churn",
    models=["logistic", "random_forest", "gradient_boosting"],
    optimize_threshold=True,
    cost_matrix=CostMatrix(cost_fn=5.0, cost_fp=1.0),
    explain=True,
    revenue_per_customer=100.0,
    intervention_cost=10.0,
)
pipeline = ChurnPipeline(config=config)
result = pipeline.run("data.csv", target="churn")

print(f"Best model: {result.best_model_name} (F1={result.best_result.f1:.4f})")
print(f"Optimal threshold: {result.threshold_result.optimal_threshold:.4f}")
```

### Threshold Optimization

```python
from churnguard.threshold import ThresholdOptimizer, CostMatrix

optimizer = ThresholdOptimizer()

# F1-optimal threshold
result = optimizer.optimize(y_true, y_proba, strategy="f1")

# Youden's J statistic
result = optimizer.optimize(y_true, y_proba, strategy="youden")

# Cost-sensitive (asymmetric FN/FP costs)
result = optimizer.optimize(
    y_true, y_proba,
    strategy="cost_sensitive",
    cost_matrix=CostMatrix(cost_fn=5.0, cost_fp=1.0),
)
```

### Explainability

```python
from churnguard.explainability import ChurnExplainer

explainer = ChurnExplainer(model=trained_model)
explainer.fit(X_train, feature_names=feature_names)

# Global feature importance
global_exp = explainer.explain_global(X_test)
print(global_exp.feature_importance)

# Per-customer explanation
customer_exp = explainer.explain_customer(X_test, customer_index=0)
print(customer_exp.feature_contribution)
```

## Configuration

Create a `churnguard.yaml`:

```yaml
features:
  numeric_impute_strategy: median
  categorical_max_cardinality: 20
  scaling: standard

models:
  logistic:
    C: 1.0
    max_iter: 1000
  random_forest:
    n_estimators: 200
    max_depth: 15
  gradient_boosting:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 5

evaluation:
  cv_folds: 5
  primary_metric: f1
  random_state: 42
```

## License

MIT
