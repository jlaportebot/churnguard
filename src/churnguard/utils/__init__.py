"""Utility functions for ChurnGuard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging for ChurnGuard.

    Parameters
    ----------
    level : str
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    log_file : Path, optional
        Path to a log file. If None, logs to stderr only.
    """
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Parameters
    ----------
    base : dict
        Base configuration.
    override : dict
        Override configuration (takes precedence).

    Returns
    -------
    dict
        Merged configuration.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


DEFAULT_CONFIG: dict[str, Any] = {
    "features": {
        "numeric_impute_strategy": "median",
        "scaling": "standard",
        "categorical_max_cardinality": 20,
        "generate_interactions": True,
        "generate_polynomials": False,
    },
    "models": {
        "logistic": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "gradient_boosting": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
        },
    },
    "evaluation": {
        "cv_folds": 5,
        "primary_metric": "f1",
        "random_state": 42,
    },
    "output": {
        "save_plots": True,
        "save_json": True,
        "plot_format": "png",
        "dpi": 150,
    },
}


def get_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Get the effective configuration by merging defaults with a user config.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to user config file. If None, returns defaults.

    Returns
    -------
    dict
        Effective configuration.
    """
    config = DEFAULT_CONFIG.copy()
    if config_path:
        try:
            user_config = load_config(config_path)
            config = merge_configs(config, user_config)
        except FileNotFoundError:
            logger.warning("Config file %s not found. Using defaults.", config_path)
    return config


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        The directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate_dict(d: dict[str, Any], max_items: int = 10) -> dict[str, Any]:
    """Truncate a dictionary for display, keeping top items by value.

    Parameters
    ----------
    d : dict
        Dictionary to truncate (values must be numeric).
    max_items : int
        Maximum items to keep.

    Returns
    -------
    dict
        Truncated dictionary.
    """
    if len(d) <= max_items:
        return d
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:max_items]
    return dict(sorted_items)
