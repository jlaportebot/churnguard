# ChurnGuard

Customer churn prediction CLI and library with ML model comparison.

## Features

- **CLI tool** for quick churn analysis from the command line
- **Feature engineering pipeline** with automatic preprocessing
- **Multiple ML models**: Logistic Regression, Random Forest, Gradient Boosting
- **Model evaluation & comparison** with rich metrics output
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

# Compare all models
churnguard compare data.csv --target churn

# Generate sample data for testing
churnguard sample --output sample_data.csv --rows 1000

# Score new customers
churnguard score new_customers.csv --model results/best_model.joblib
```

### Python API

```python
from churnguard.data import DataLoader
from churnguard.features import FeatureEngineer
from churnguard.models import ModelRegistry
from churnguard.evaluation import ModelEvaluator

# Load and prepare data
loader = DataLoader("data.csv", target_column="churn")
X_train, X_test, y_train, y_test = loader.split()

# Engineer features
engineer = FeatureEngineer()
X_train = engineer.fit_transform(X_train)
X_test = engineer.transform(X_test)

# Train and compare models
registry = ModelRegistry()
results = registry.compare_all(X_train, X_test, y_train, y_test)

# Get best model
best = registry.get_best(results, metric="f1")
print(f"Best model: {best.name} (F1: {best.f1:.4f})")
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
