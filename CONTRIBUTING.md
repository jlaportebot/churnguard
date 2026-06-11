# Contributing to ChurnGuard

Thanks for wanting to contribute! This project is a CLI + library for churn prediction with ML model comparison, threshold optimization, and SHAP explainability. All help is welcome — code, docs, tests, bug reports, feature ideas.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/jlaportebot/churnguard.git
cd churnguard
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"

# Run tests
pytest -v

# Run linting
ruff check .
ruff format --check .

# Type check
mypy src/

# Run the CLI
churnguard --help
```

## How to Contribute

### 1. Find Something to Work On

- Check [open issues](https://github.com/jlaportebot/churnguard/issues) — look for `good first issue`, `help wanted`, `docs`
- Have an idea? Open an issue first to discuss before building
- Found a bug? Report it with the bug template

### 2. Make Your Changes

```bash
# Create a branch
git checkout -b feature/your-thing
# or
git checkout -b fix/your-thing
# or
git checkout -b docs/your-thing

# Make changes, test locally
pytest -v
ruff check .
ruff format --check .
mypy src/
```

### 3. Submit a PR

- Push your branch and open a PR against `main`
- Fill out the PR template
- Link the issue: `Fixes #123` or `Closes #123`
- CI must pass (tests, lint, format, type check)

## Code Standards

### Python Style

- **ruff** for linting + formatting — enforced in CI
- **mypy** for type checking — enforced in CI
- **Python 3.9+** — use modern syntax (union types `|`, `match`/`case`, etc.)
- Line length: 100 chars (ruff default)

### Architecture

```
src/churnguard/
├── cli.py              # Click CLI entry point
├── pipeline.py         # Main ChurnPipeline orchestration
├── config.py           # PipelineConfig, YAML loading
├── models/             # Model implementations (entry-point registry)
│   ├── base.py         # BaseChurnModel abstract class
│   ├── logistic.py
│   ├── random_forest.py
│   └── gradient_boosting.py
├── threshold.py        # ThresholdOptimizer, strategies
├── explainability.py   # ChurnExplainer, SHAP integration
├── evaluation.py       # Metrics, cross-validation
├── features.py         # FeatureEngineer, preprocessing
├── business.py         # BusinessImpactAnalyzer
├── viz.py              # Optional visualization (matplotlib/seaborn)
└── utils.py            # Helpers
```

**Key principles:**
- Models register via entry points — add new models without touching core code
- Pipeline orchestrates; individual components are testable in isolation
- CLI is thin — logic lives in library functions
- Optional deps (`viz`) don't break core if missing

### Testing

- **Unit tests**: `tests/test_*.py` — test individual functions/classes
- **Integration tests**: `tests/test_pipeline.py` — end-to-end pipeline runs
- **Fixtures**: `tests/conftest.py` — sample data, trained models
- **Coverage**: Aim for ≥80% on new code

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=churnguard --cov-report=term-missing

# Skip slow tests
pytest -m "not slow"
```

### Adding a New Model

1. Create `src/churnguard/models/your_model.py`
2. Subclass `BaseChurnModel`
3. Implement `fit()`, `predict_proba()`, `get_params()`, `set_params()`
4. Register in `pyproject.toml` under `[project.entry-points."churnguard.models"]`
5. Add tests in `tests/test_models.py`
6. Update docs/README if needed

### Adding a Threshold Strategy

1. Add strategy to `ThresholdOptimizer.optimize()` in `threshold.py`
2. Implement the logic (usually a one-liner calling sklearn/scipy)
3. Add tests
4. Document in CLI help and README

## PR Requirements

- [ ] Tests pass (`pytest -v`)
- [ ] Lint clean (`ruff check .`)
- [ ] Format clean (`ruff format --check .`)
- [ ] Type check clean (`mypy src/`)
- [ ] No debug code left (`print()`, `breakpoint()`, commented-out code)
- [ ] Docstrings on public functions/classes (Google style)
- [ ] New code has tests
- [ ] Updated docs if user-facing behavior changed

## Communication

- **Issues** — Bugs, features, questions
- **Discussions** — General chat, ideas, "how do I...?"
- **PRs** — Code review happens here

Response time: usually within a day or two.

## Recognition

All contributors get added to the contributors list in releases. We value code, docs, tests, bug reports, triage, and answering questions.

---

**Questions?** Open a [Discussion](https://github.com/jlaportebot/churnguard/discussions) or comment on an issue.