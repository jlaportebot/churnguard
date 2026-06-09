"""Data loading and preprocessing module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Common column name variations for the target
TARGET_ALIASES = {
    "churn",
    "churned",
    "churn_flag",
    "is_churned",
    "exited",
    "has_churned",
    "attrition",
    "left",
    "target",
    "label",
}

# Columns that are typically identifiers and should be dropped
ID_COLUMN_PATTERNS = {
    "customerid",
    "customer_id",
    "user_id",
    "userid",
    "id",
    "row_id",
    "rownumber",
    "surname",
    "name",
}


class DataValidationError(Exception):
    """Raised when input data fails validation."""


class DataLoader:
    """Load, validate, and split churn datasets.

    Parameters
    ----------
    source : str or Path
        Path to a CSV file.
    target_column : str, optional
        Name of the target column. If None, auto-detection is attempted.
    id_columns : list of str, optional
        Columns to drop as identifiers. Auto-detected if None.
    drop_na_target : bool
        Whether to drop rows with missing target values.
    random_state : int
        Random state for reproducibility.
    test_size : float
        Fraction of data to use as test set.
    """

    def __init__(
        self,
        source: str | Path,
        target_column: str | None = None,
        id_columns: list[str] | None = None,
        drop_na_target: bool = True,
        random_state: int = 42,
        test_size: float = 0.2,
    ):
        self.source = Path(source)
        self.target_column = target_column
        self.id_columns = id_columns
        self.drop_na_target = drop_na_target
        self.random_state = random_state
        self.test_size = test_size
        self._df: pd.DataFrame | None = None
        self._target_column_resolved: str | None = None
        self._id_columns_resolved: list[str] | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded DataFrame."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load data from source file."""
        if not self.source.exists():
            raise FileNotFoundError(f"Data file not found: {self.source}")

        suffix = self.source.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.source)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(self.source)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(self.source)
        else:
            raise DataValidationError(
                f"Unsupported file format: {suffix}. Use .csv, .parquet, or .xlsx."
            )

        logger.info("Loaded %d rows and %d columns from %s", len(df), len(df.columns), self.source)
        return df

    def validate(self) -> DataLoader:
        """Validate the loaded data.

        Returns
        -------
        self : DataLoader
            For method chaining.

        Raises
        ------
        DataValidationError
            If data fails validation checks.
        """
        df = self.df

        if df.empty:
            raise DataValidationError("Dataset is empty.")

        if len(df) < 10:
            logger.warning("Dataset has fewer than 10 rows; results may be unreliable.")

        # Resolve target column
        self._resolve_target(df)

        # Validate target values
        df[self._target_column_resolved]
        if self.drop_na_target:
            n_before = len(df)
            self._df = df.dropna(subset=[self._target_column_resolved])
            n_dropped = n_before - len(self._df)
            if n_dropped > 0:
                logger.info("Dropped %d rows with missing target values.", n_dropped)
            df = self._df

        unique_targets = df[self._target_column_resolved].nunique()
        if unique_targets < 2:
            raise DataValidationError(
                f"Target column '{self._target_column_resolved}' has only "
                f"{unique_targets} unique value(s). Need at least 2 for classification."
            )

        if unique_targets > 20:
            logger.warning(
                "Target column has %d unique values — this may be a regression target, "
                "not a classification target.",
                unique_targets,
            )

        # Resolve ID columns
        self._resolve_id_columns(df)

        logger.info("Validation passed. Target: '%s'", self._target_column_resolved)
        return self

    def _resolve_target(self, df: pd.DataFrame) -> None:
        """Resolve the target column name."""
        if self.target_column is not None:
            if self.target_column not in df.columns:
                raise DataValidationError(
                    f"Specified target column '{self.target_column}' not found "
                    f"in data. Available: {list(df.columns)}"
                )
            self._target_column_resolved = self.target_column
            return

        # Auto-detect target column
        for col in df.columns:
            if col.lower() in TARGET_ALIASES:
                self._target_column_resolved = col
                logger.info("Auto-detected target column: '%s'", col)
                return

        # Try binary columns (2 unique values) as last resort
        binary_cols = [c for c in df.columns if df[c].nunique() == 2]
        if len(binary_cols) == 1:
            self._target_column_resolved = binary_cols[0]
            logger.info("Auto-detected single binary column as target: '%s'", binary_cols[0])
            return

        raise DataValidationError(
            f"Could not auto-detect target column. Please specify --target. "
            f"Available columns: {list(df.columns)}"
        )

    def _resolve_id_columns(self, df: pd.DataFrame) -> None:
        """Resolve columns that should be dropped as identifiers."""
        if self.id_columns is not None:
            self._id_columns_resolved = self.id_columns
            return

        auto_ids = []
        for col in df.columns:
            if col.lower() in ID_COLUMN_PATTERNS and col != self._target_column_resolved:
                auto_ids.append(col)

        self._id_columns_resolved = auto_ids
        if auto_ids:
            logger.info("Auto-detected ID columns to drop: %s", auto_ids)

    def get_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into feature matrix X and target vector y.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (ID and target columns removed).
        y : pd.Series
            Target vector.
        """
        df = self.df
        if self._target_column_resolved is None:
            self.validate()

        target = self._target_column_resolved
        ids = self._id_columns_resolved or []

        drop_cols = [target] + [c for c in ids if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[target]

        # Encode target if it's not numeric
        if not pd.api.types.is_numeric_dtype(y):
            unique_vals = sorted(y.dropna().unique())
            if len(unique_vals) == 2:
                label_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                y = y.map(label_map).astype(int)
                logger.info("Encoded target: %s → 0, %s → 1", unique_vals[0], unique_vals[1])
            else:
                raise DataValidationError(
                    f"Non-numeric target with {len(unique_vals)} unique values "
                    f"is not supported for binary classification."
                )

        return X, y

    def split(
        self, test_size: float | None = None, random_state: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Parameters
        ----------
        test_size : float, optional
            Override the test_size from constructor.
        random_state : int, optional
            Override the random_state from constructor.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        self.validate()
        X, y = self.get_features_and_target()

        ts = test_size if test_size is not None else self.test_size
        rs = random_state if random_state is not None else self.random_state

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=rs, stratify=y
        )

        logger.info(
            "Split: %d train, %d test (churn rate: %.1f%% / %.1f%%)",
            len(X_train),
            len(X_test),
            y_train.mean() * 100,
            y_test.mean() * 100,
        )

        return X_train, X_test, y_train, y_test

    @property
    def target_name(self) -> str:
        """Return the resolved target column name."""
        if self._target_column_resolved is None:
            self.validate()
        return self._target_column_resolved

    @property
    def detected_id_columns(self) -> list[str]:
        """Return the resolved ID columns."""
        if self._id_columns_resolved is None:
            self.validate()
        return self._id_columns_resolved or []


def generate_sample_data(
    n_rows: int = 1000,
    churn_rate: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic churn data for testing and demos.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    churn_rate : float
        Approximate churn rate (0.0 to 1.0).
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with a 'churn' column.
    """
    rng = np.random.RandomState(random_state)

    # Customer demographics
    tenure = rng.exponential(scale=36, size=n_rows).astype(int).clip(1, 120)
    monthly_charges = rng.normal(loc=65, scale=30, size=n_rows).clip(18, 150)
    total_charges = tenure * monthly_charges + rng.normal(0, 100, size=n_rows)

    # Categorical features
    contract_types = ["Month-to-month", "One year", "Two year"]
    contract_weights = [0.55, 0.25, 0.20]
    contract = rng.choice(contract_types, size=n_rows, p=contract_weights)

    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_weights = [0.35, 0.20, 0.25, 0.20]
    payment_method = rng.choice(payment_methods, size=n_rows, p=payment_weights)

    internet_services = ["DSL", "Fiber optic", "No"]
    internet_weights = [0.35, 0.45, 0.20]
    internet_service = rng.choice(internet_services, size=n_rows, p=internet_weights)

    # Binary features
    senior_citizen = rng.binomial(1, 0.16, size=n_rows)
    partner = rng.binomial(1, 0.48, size=n_rows)
    dependents = rng.binomial(1, 0.30, size=n_rows)
    online_security = rng.binomial(1, 0.35, size=n_rows)
    tech_support = rng.binomial(1, 0.30, size=n_rows)
    streaming_tv = rng.binomial(1, 0.40, size=n_rows)

    # Number of services
    n_services = online_security + tech_support + streaming_tv + partner + dependents

    # Compute churn probability based on features
    log_odds = (
        -2.5
        + 0.8 * (contract == "Month-to-month")
        - 0.5 * (contract == "Two year")
        + 0.6 * (internet_service == "Fiber optic")
        - 0.3 * (internet_service == "No")
        + 0.4 * (payment_method == "Electronic check")
        - 0.02 * tenure
        + 0.01 * monthly_charges
        - 0.15 * n_services
        + 0.3 * senior_citizen
        - 0.2 * partner
        - 0.25 * dependents
        - 0.4 * online_security
        - 0.35 * tech_support
    )

    # Adjust to match desired churn rate
    current_rate = 1 / (1 + np.exp(-log_odds)).mean()
    adjustment = np.log(churn_rate / (1 - churn_rate)) - np.log(current_rate / (1 - current_rate))
    log_odds += adjustment

    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n_rows)],
            "tenure": tenure,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": np.round(total_charges, 2).clip(0),
            "contract": contract,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "n_services": n_services,
            "churn": churn,
        }
    )

    # Inject some missing values (5% of numeric cols)
    for col in ["tenure", "monthly_charges", "total_charges"]:
        mask = rng.random(n_rows) < 0.05
        df.loc[mask, col] = np.nan

    actual_rate = df["churn"].mean()
    logger.info("Generated %d rows with %.1f%% churn rate", n_rows, actual_rate * 100)

    return df
