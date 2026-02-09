#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/xjtu-sy-bearing
#
#  This work is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the condition that the above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  For more information, visit: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   xjtu-sy-bearing onset and RUL prediction ML Pipeline

"""LightGBM baseline model for RUL prediction.

This module implements a gradient boosting baseline using LightGBM for Remaining
Useful Life (RUL) prediction on handcrafted features. It provides a strong,
interpretable baseline that's difficult to fake with simple heuristics.

Features:
- Hyperparameter configuration via dataclass
- Cross-validation training with leave-one-bearing-out
- Feature importance extraction (gain-based and split-based)
- SHAP value analysis for interpretability
- Integration with existing CV and metrics infrastructure

Reference:
    Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
    NeurIPS 2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm") from e

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class LGBMConfig:
    """Configuration for LightGBM RUL prediction model.

    Attributes:
        objective: LightGBM objective function.
        metric: Evaluation metric.
        num_leaves: Maximum number of leaves in one tree.
        learning_rate: Boosting learning rate.
        n_estimators: Number of boosting iterations.
        max_depth: Maximum depth of tree (-1 for no limit).
        min_child_samples: Minimum number of data in one leaf.
        subsample: Fraction of training data to use.
        colsample_bytree: Fraction of features to use per tree.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads (-1 for all).
        verbose: Verbosity level.
        early_stopping_rounds: Stop if no improvement for this many rounds.
    """

    objective: str = "regression"
    metric: str = "rmse"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    early_stopping_rounds: int = 50

    def to_lgb_params(self) -> dict[str, Any]:
        """Convert to LightGBM parameter dictionary.

        Returns:
            Dictionary of LightGBM parameters.
        """
        return {
            "objective": self.objective,
            "metric": self.metric,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }


@dataclass
class LGBMPrediction:
    """Result of LightGBM RUL prediction.

    Attributes:
        y_pred: Predicted RUL values.
        y_true: Ground truth RUL values.
        bearing_ids: Bearing IDs for each sample.
        feature_names: Names of features used.
        feature_importance_gain: Feature importance by gain.
        feature_importance_split: Feature importance by split count.
    """

    y_pred: np.ndarray
    y_true: np.ndarray
    bearing_ids: np.ndarray
    feature_names: list[str]
    feature_importance_gain: np.ndarray | None = None
    feature_importance_split: np.ndarray | None = None


@dataclass
class CVResult:
    """Result of cross-validation training.

    Attributes:
        fold_id: Fold identifier.
        train_rmse: Training RMSE.
        val_rmse: Validation RMSE.
        val_mae: Validation MAE.
        val_bearing: Held-out bearing ID(s).
        y_pred: Validation predictions.
        y_true: Validation ground truth.
        best_iteration: Best iteration (if early stopping).
        feature_importance_gain: Feature importance by gain.
        feature_importance_split: Feature importance by split count.
    """

    fold_id: int
    train_rmse: float
    val_rmse: float
    val_mae: float
    val_bearing: list[str]
    y_pred: np.ndarray
    y_true: np.ndarray
    best_iteration: int | None = None
    feature_importance_gain: np.ndarray | None = None
    feature_importance_split: np.ndarray | None = None


class LightGBMBaseline:
    """LightGBM regression model for RUL prediction.

    This baseline uses handcrafted time-domain and frequency-domain features
    to predict Remaining Useful Life. It provides interpretable predictions
    through feature importance and SHAP values.

    Example:
        >>> from src.models.baselines import LightGBMBaseline, LGBMConfig
        >>> config = LGBMConfig(n_estimators=200, learning_rate=0.1)
        >>> model = LightGBMBaseline(config)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: LGBMConfig | None = None) -> None:
        """Initialize LightGBM baseline.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or LGBMConfig()
        self._model: lgb.LGBMRegressor | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> "LightGBMBaseline":
        """Fit the LightGBM model.

        Args:
            X_train: Training features.
            y_train: Training targets (RUL values).
            X_val: Validation features (optional, for early stopping).
            y_val: Validation targets (optional).
            feature_names: Names of features (inferred from DataFrame if not provided).

        Returns:
            Self for method chaining.
        """
        # Extract feature names
        if feature_names is not None:
            self._feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()
        else:
            self._feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Convert to numpy if DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Initialize model
        self._model = lgb.LGBMRegressor(**self.config.to_lgb_params())

        # Fit with or without validation set
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=self.config.verbose > 0,
                )
            ]

        self._model.fit(X_train, y_train, **fit_params)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict RUL for given features.

        Args:
            X: Features to predict on.

        Returns:
            Predicted RUL values (clipped to non-negative).

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = self._model.predict(X)
        # RUL should be non-negative
        return np.maximum(predictions, 0.0)

    def get_feature_importance(
        self,
        importance_type: Literal["gain", "split"] = "gain",
    ) -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance ("gain" or "split").

        Returns:
            DataFrame with feature names and importance values, sorted by importance.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before getting importance.")

        importance = self._model.feature_importances_
        if importance_type == "split":
            importance = self._model.booster_.feature_importance(importance_type="split")

        df = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_shap_values(
        self,
        X: np.ndarray | pd.DataFrame,
        max_samples: int = 1000,
    ) -> tuple[np.ndarray, shap.Explainer | None]:
        """Compute SHAP values for interpretability.

        Args:
            X: Features to explain.
            max_samples: Maximum samples to use for SHAP (for efficiency).

        Returns:
            Tuple of (SHAP values array, SHAP explainer object).

        Raises:
            RuntimeError: If model is not fitted.
            ImportError: If SHAP is not installed.
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")

        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before computing SHAP values.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Subsample for efficiency
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)

        return shap_values, explainer

    def save(self, path: str | Path) -> None:
        """Save model to file.

        Args:
            path: Path to save the model.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before saving.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path))

    def load(self, path: str | Path) -> "LightGBMBaseline":
        """Load model from file.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        self._model = lgb.LGBMRegressor()
        self._model._Booster = lgb.Booster(model_file=str(path))
        self._is_fitted = True
        return self

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names

    @property
    def best_iteration(self) -> int | None:
        """Get best iteration (if early stopping was used)."""
        if self._model is None:
            return None
        return getattr(self._model, "best_iteration_", None)


def train_with_cv(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul",
    bearing_col: str = "bearing_id",
    cv_split: "CVSplit | None" = None,
    config: LGBMConfig | None = None,
    verbose: bool = True,
) -> tuple[list[CVResult], pd.DataFrame]:
    """Train LightGBM with cross-validation.

    Args:
        features_df: DataFrame with features and metadata.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        bearing_col: Name of bearing ID column.
        cv_split: Cross-validation split (uses leave-one-bearing-out if None).
        config: Model configuration.
        verbose: Print progress.

    Returns:
        Tuple of (list of CVResult, aggregated feature importance DataFrame).
    """
    from src.training.cv import leave_one_bearing_out
    from src.training.metrics import rmse, mae

    if cv_split is None:
        cv_split = leave_one_bearing_out(features_df)

    config = config or LGBMConfig()
    results = []
    importance_dfs = []

    for fold in cv_split:
        # Get train/val data
        train_df = features_df.iloc[fold.train_indices]
        val_df = features_df.iloc[fold.val_indices]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        # Train model
        model = LightGBMBaseline(config)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Compute metrics
        train_rmse = rmse(y_train, y_pred_train)
        val_rmse = rmse(y_val, y_pred_val)
        val_mae_val = mae(y_val, y_pred_val)

        # Get feature importance
        importance_df = model.get_feature_importance("gain")
        importance_df["fold_id"] = fold.fold_id
        importance_dfs.append(importance_df)

        result = CVResult(
            fold_id=fold.fold_id,
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            val_mae=val_mae_val,
            val_bearing=fold.val_bearings,
            y_pred=y_pred_val,
            y_true=y_val,
            best_iteration=model.best_iteration,
            feature_importance_gain=importance_df["importance"].values,
            feature_importance_split=None,
        )
        results.append(result)

        if verbose:
            print(
                f"Fold {fold.fold_id:2d} | "
                f"Val bearing: {fold.val_bearings[0]:12s} | "
                f"Train RMSE: {train_rmse:6.2f} | "
                f"Val RMSE: {val_rmse:6.2f} | "
                f"Val MAE: {val_mae_val:6.2f}"
            )

    # Aggregate feature importance across folds
    all_importance = pd.concat(importance_dfs, ignore_index=True)
    agg_importance = (
        all_importance.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    agg_importance.columns = ["feature", "importance_mean", "importance_std"]

    if verbose:
        print("\n" + "=" * 60)
        mean_val_rmse = np.mean([r.val_rmse for r in results])
        std_val_rmse = np.std([r.val_rmse for r in results])
        print(f"CV Results: Val RMSE = {mean_val_rmse:.2f} +/- {std_val_rmse:.2f}")
        print("=" * 60)

    return results, agg_importance


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Get feature column names from DataFrame.

    Automatically excludes metadata columns.

    Args:
        df: Features DataFrame.
        exclude_cols: Additional columns to exclude.

    Returns:
        List of feature column names.
    """
    metadata_cols = {
        "condition",
        "bearing_id",
        "filename",
        "file_idx",
        "total_files",
        "rul",
        "rul_original",
        "rul_twostage",
        "is_post_onset",
    }
    if exclude_cols:
        metadata_cols.update(exclude_cols)

    return [col for col in df.columns if col not in metadata_cols]


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    title: str = "Feature Importance (LightGBM)",
) -> "plt.Figure":
    """Plot feature importance as horizontal bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance_mean' columns.
        top_n: Number of top features to show.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features["importance_mean"].values, xerr=top_features.get("importance_std", 0))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
) -> None:
    """Plot SHAP summary plot.

    Args:
        shap_values: SHAP values array.
        X: Feature matrix.
        feature_names: Names of features.
        max_display: Maximum features to display.

    Raises:
        ImportError: If SHAP is not installed.
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is required. Install with: pip install shap")

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values

    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display)


def evaluate_lgbm_cv(
    cv_results: list[CVResult],
    features_df: pd.DataFrame,
    bearing_col: str = "bearing_id",
) -> dict[str, Any]:
    """Evaluate cross-validation results.

    Args:
        cv_results: List of CVResult from cross-validation.
        features_df: Original features DataFrame.
        bearing_col: Name of bearing ID column.

    Returns:
        Dictionary with aggregate metrics and per-bearing breakdown.
    """
    from src.training.metrics import (
        evaluate_predictions,
        per_bearing_metrics,
        aggregate_bearing_metrics,
    )

    # Aggregate all predictions
    all_y_true = np.concatenate([r.y_true for r in cv_results])
    all_y_pred = np.concatenate([r.y_pred for r in cv_results])

    # Get bearing IDs for all validation samples
    all_bearing_ids = []
    for r in cv_results:
        # All samples in this fold belong to the val_bearing
        all_bearing_ids.extend([r.val_bearing[0]] * len(r.y_true))
    all_bearing_ids = np.array(all_bearing_ids)

    # Overall metrics
    overall = evaluate_predictions(all_y_true, all_y_pred)

    # Per-bearing metrics
    per_bearing_df = per_bearing_metrics(all_y_true, all_y_pred, all_bearing_ids)

    # Summary across bearings
    summaries = aggregate_bearing_metrics(per_bearing_df)

    # CV fold statistics
    cv_stats = {
        "val_rmse_mean": np.mean([r.val_rmse for r in cv_results]),
        "val_rmse_std": np.std([r.val_rmse for r in cv_results]),
        "val_mae_mean": np.mean([r.val_mae for r in cv_results]),
        "val_mae_std": np.std([r.val_mae for r in cv_results]),
        "train_rmse_mean": np.mean([r.train_rmse for r in cv_results]),
    }

    return {
        "overall": overall,
        "per_bearing": per_bearing_df,
        "summaries": summaries,
        "cv_stats": cv_stats,
    }
