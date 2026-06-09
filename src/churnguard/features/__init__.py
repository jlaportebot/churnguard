"""Feature engineering pipeline for churn prediction."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Default cardinality threshold for categorical encoding
DEFAULT_MAX_CARDINALITY = 20


class FeatureEngineer:
    """Automatic feature engineering for churn datasets.

    Handles:
    - Numeric feature imputation and scaling
    - Categorical feature encoding (one-hot for low cardinality)
    - High-cardinality categorical features (frequency encoding)
    - Interaction feature generation
    - Polynomial feature generation (optional)

    Parameters
    ----------
    numeric_impute_strategy : str
        Imputation strategy for numeric columns: 'mean', 'median', or 'most_frequent'.
    scaling : str
        Scaling method: 'standard', 'minmax', or 'none'.
    categorical_max_cardinality : int
        Maximum number of unique values for one-hot encoding.
        Categories with more values use frequency encoding.
    generate_interactions : bool
        Whether to generate interaction features between top numeric columns.
    generate_polynomials : bool
        Whether to generate polynomial features for numeric columns.
    polynomial_degree : int
        Degree for polynomial features (if enabled).
    """

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        scaling: str = "standard",
        categorical_max_cardinality: int = DEFAULT_MAX_CARDINALITY,
        generate_interactions: bool = True,
        generate_polynomials: bool = False,
        polynomial_degree: int = 2,
    ):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.scaling = scaling
        self.categorical_max_cardinality = categorical_max_cardinality
        self.generate_interactions = generate_interactions
        self.generate_polynomials = generate_polynomials
        self.polynomial_degree = polynomial_degree

        self._column_transformer: Optional[ColumnTransformer] = None
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._onehot_cols: list[str] = []
        self._freq_cols: list[str] = []
        self._freq_maps: dict[str, dict] = {}
        self._interaction_pairs: list[tuple[str, str]] = []
        self._is_fitted = False

    def _classify_columns(self, X: pd.DataFrame) -> None:
        """Classify columns into numeric and categorical."""
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Split categorical by cardinality
        self._onehot_cols = []
        self._freq_cols = []
        for col in self._categorical_cols:
            n_unique = X[col].nunique()
            if n_unique <= self.categorical_max_cardinality:
                self._onehot_cols.append(col)
            else:
                self._freq_cols.append(col)
                logger.info(
                    "Using frequency encoding for '%s' (%d unique values)",
                    col,
                    n_unique,
                )

    def _compute_frequency_maps(self, X: pd.DataFrame) -> None:
        """Compute frequency maps for high-cardinality categorical columns."""
        self._freq_maps = {}
        for col in self._freq_cols:
            freq = X[col].value_counts(normalize=True).to_dict()
            self._freq_maps[col] = freq

    def _apply_frequency_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding to high-cardinality columns."""
        X = X.copy()
        for col in self._freq_cols:
            if col in self._freq_maps:
                X[col] = X[col].map(self._freq_maps[col]).fillna(0)
        return X

    def _select_interaction_pairs(self, X: pd.DataFrame) -> None:
        """Select top numeric column pairs for interaction features."""
        if not self.generate_interactions or len(self._numeric_cols) < 2:
            self._interaction_pairs = []
            return

        # Compute correlations with each other
        corr = X[self._numeric_cols].corr().abs()
        corr_values = corr.values.copy()
        np.fill_diagonal(corr_values, 0)

        # Pick top pairs by correlation (max 5 interactions)
        pairs = []
        for _ in range(min(5, len(self._numeric_cols) * (len(self._numeric_cols) - 1) // 2)):
            if corr_values.size == 0:
                break
            max_flat = np.argmax(corr_values)
            i, j = np.unravel_index(max_flat, corr_values.shape)
            if i != j:
                col_i = self._numeric_cols[i]
                col_j = self._numeric_cols[j]
                pairs.append((col_i, col_j))
                corr_values[i, j] = 0
                corr_values[j, i] = 0
            else:
                break

        self._interaction_pairs = pairs
        if pairs:
            logger.info("Generated %d interaction feature pairs", len(pairs))

    def _build_column_transformer(self) -> ColumnTransformer:
        """Build the sklearn ColumnTransformer for preprocessing."""
        transformers = []

        # Numeric pipeline
        if self._numeric_cols:
            num_steps = [("imputer", SimpleImputer(strategy=self.numeric_impute_strategy))]
            if self.scaling == "standard":
                num_steps.append(("scaler", StandardScaler()))
            num_pipeline = Pipeline(num_steps)
            transformers.append(("numeric", num_pipeline, self._numeric_cols))

        # One-hot encoding pipeline
        if self._onehot_cols:
            cat_pipeline = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            drop="if_binary",
                        ),
                    ),
                ]
            )
            transformers.append(("categorical", cat_pipeline, self._onehot_cols))

        # Frequency-encoded columns are already numeric, add simple imputer
        if self._freq_cols:
            freq_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
            )
            transformers.append(("freq_encoded", freq_pipeline, self._freq_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> FeatureEngineer:
        """Fit the feature engineering pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series, optional
            Target (not used in feature engineering, but accepted for sklearn compat).

        Returns
        -------
        self
        """
        self._classify_columns(X)
        self._compute_frequency_maps(X)

        # Apply frequency encoding before building transformer
        X_processed = self._apply_frequency_encoding(X)

        # Select interaction pairs
        self._select_interaction_pairs(X_processed)

        # Build and fit column transformer
        self._column_transformer = self._build_column_transformer()
        self._column_transformer.fit(X_processed, y)

        self._is_fitted = True
        logger.info(
            "Fitted feature pipeline: %d numeric, %d onehot, %d freq-encoded columns",
            len(self._numeric_cols),
            len(self._onehot_cols),
            len(self._freq_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")

        X_processed = self._apply_frequency_encoding(X)

        # Apply column transformer
        X_transformed = self._column_transformer.transform(X_processed)

        # Get feature names
        try:
            feature_names = list(self._column_transformer.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        result = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        # Add interaction features
        if self._interaction_pairs:
            interactions = self._add_interactions(X_processed)
            result = pd.concat([result, interactions], axis=1)

        # Add polynomial features
        if self.generate_polynomials and self._numeric_cols:
            polys = self._add_polynomials(X_processed)
            result = pd.concat([result, polys], axis=1)

        return result

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series, optional
            Target vector.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix.
        """
        return self.fit(X, y).transform(X)

    def _add_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between selected column pairs."""
        interaction_data = {}
        for col_a, col_b in self._interaction_pairs:
            if col_a in X.columns and col_b in X.columns:
                name = f"{col_a}_x_{col_b}"
                interaction_data[name] = X[col_a].fillna(0) * X[col_b].fillna(0)

        if interaction_data:
            return pd.DataFrame(interaction_data, index=X.index)
        return pd.DataFrame(index=X.index)

    def _add_polynomials(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for numeric columns."""
        poly_data = {}
        cols = [c for c in self._numeric_cols if c in X.columns]

        for col in cols:
            if self.polynomial_degree >= 2:
                poly_data[f"{col}_sq"] = X[col] ** 2
            if self.polynomial_degree >= 3:
                poly_data[f"{col}_cube"] = X[col] ** 3

        if poly_data:
            return pd.DataFrame(poly_data, index=X.index)
        return pd.DataFrame(index=X.index)

    @property
    def feature_names(self) -> list[str]:
        """Return the output feature names after transformation."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted first.")
        names = list(self._column_transformer.get_feature_names_out())
        for col_a, col_b in self._interaction_pairs:
            names.append(f"{col_a}_x_{col_b}")
        if self.generate_polynomials:
            for col in self._numeric_cols:
                if self.polynomial_degree >= 2:
                    names.append(f"{col}_sq")
                if self.polynomial_degree >= 3:
                    names.append(f"{col}_cube")
        return names

    @property
    def n_features_out(self) -> int:
        """Return the number of output features."""
        return len(self.feature_names)


class FeatureSelector:
    """Select most important features based on model or statistical criteria.

    Parameters
    ----------
    method : str
        Selection method: 'variance', 'correlation', or 'model'.
    threshold : float
        Threshold for selection (meaning depends on method).
    max_features : int, optional
        Maximum number of features to select.
    """

    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.01,
        max_features: Optional[int] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self._selected_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> FeatureSelector:
        """Fit the selector to determine which features to keep."""
        if self.method == "variance":
            variances = X.var()
            self._selected_features = variances[variances > self.threshold].index.tolist()

        elif self.method == "correlation" and y is not None:
            correlations = X.corrwith(y).abs()
            correlations = correlations.sort_values(ascending=False)
            self._selected_features = correlations[correlations > self.threshold].index.tolist()

        elif self.method == "model":
            # Model-based selection requires y
            if y is None:
                raise ValueError("y is required for model-based feature selection.")
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            importances = importances.sort_values(ascending=False)
            self._selected_features = importances[importances > self.threshold].index.tolist()

        else:
            self._selected_features = X.columns.tolist()

        if self.max_features and len(self._selected_features) > self.max_features:
            self._selected_features = self._selected_features[: self.max_features]

        logger.info(
            "Selected %d / %d features using '%s' method",
            len(self._selected_features),
            len(X.columns),
            self.method,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X to keep only selected features."""
        return X[[c for c in self._selected_features if c in X.columns]]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def selected_features(self) -> list[str]:
        """Return the list of selected feature names."""
        return self._selected_features
