"""ChurnGuard — Customer churn prediction CLI and library."""

__version__ = "0.2.0"


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency at module load time."""
    _lazy = {
        "ThresholdOptimizer": "churnguard.threshold",
        "CostMatrix": "churnguard.threshold",
        "ThresholdResult": "churnguard.threshold",
        "optimize_threshold": "churnguard.threshold",
        "ChurnExplainer": "churnguard.explainability",
        "GlobalExplanation": "churnguard.explainability",
        "CustomerExplanation": "churnguard.explainability",
        "ChurnPipeline": "churnguard.pipeline",
        "PipelineConfig": "churnguard.pipeline",
        "PipelineResult": "churnguard.pipeline",
        "compute_business_impact": "churnguard.pipeline",
        "run_pipeline": "churnguard.pipeline",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'churnguard' has no attribute {name!r}")


__all__ = [
    "ThresholdOptimizer",
    "CostMatrix",
    "ThresholdResult",
    "optimize_threshold",
    "ChurnExplainer",
    "GlobalExplanation",
    "CustomerExplanation",
    "ChurnPipeline",
    "PipelineConfig",
    "PipelineResult",
    "compute_business_impact",
    "run_pipeline",
]
