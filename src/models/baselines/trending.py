"""Statistical trending baseline for RUL prediction.

This module implements simple signal processing baselines for Remaining Useful Life
(RUL) prediction, serving as a reference point for more complex machine learning models.

The baselines use health indicators (HI) that naturally trend as bearings degrade:
- RMS: Root Mean Square amplitude increases with degradation
- Kurtosis: Statistical measure of impulsiveness, spikes during fault development

These approaches require no machine learning - just signal processing and curve fitting.

Reference:
    Wang, Y., et al. "A data-driven health indicator and remaining useful life prediction
    method based on LSTM and CNN." Mechanical Systems and Signal Processing, 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TrendingPrediction:
    """Result of a trending-based RUL prediction.

    Attributes:
        y_pred: Predicted RUL values.
        y_true: Ground truth RUL values (if available).
        bearing_id: Identifier for the bearing.
        method: Prediction method used.
        health_indicator: Health indicator values used for prediction.
        threshold: Failure threshold used.
    """

    y_pred: np.ndarray
    y_true: np.ndarray | None
    bearing_id: str
    method: str
    health_indicator: np.ndarray
    threshold: float | None = None


class RMSThresholdBaseline:
    """RUL estimation based on RMS trending with threshold detection.

    This baseline assumes that bearing failure occurs when RMS amplitude
    exceeds a threshold. RUL is estimated by fitting a trend line to RMS
    values and extrapolating to the threshold.

    The approach:
    1. Calculate RMS for each sample (or use pre-computed features)
    2. Fit a linear/exponential trend to the RMS values
    3. Extrapolate to estimate time-to-threshold (RUL)
    """

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        trend_type: Literal["linear", "exponential"] = "linear",
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the RMS threshold baseline.

        Args:
            threshold_percentile: Percentile of RMS values to use as failure threshold.
                Uses the max RMS from training data at this percentile.
            trend_type: Type of trend fitting ("linear" or "exponential").
            smoothing_window: Window size for moving average smoothing.
        """
        self.threshold_percentile = threshold_percentile
        self.trend_type = trend_type
        self.smoothing_window = smoothing_window
        self._threshold: float | None = None
        self._fitted = False

    def fit(self, features_df: pd.DataFrame, rms_column: str = "h_rms") -> "RMSThresholdBaseline":
        """Fit the threshold from training data.

        Args:
            features_df: DataFrame with features and metadata.
            rms_column: Column name for RMS values.

        Returns:
            Self for method chaining.
        """
        # Get final RMS values (at failure) for each bearing
        final_rms_values = []
        for bearing_id in features_df["bearing_id"].unique():
            bearing_data = features_df[features_df["bearing_id"] == bearing_id]
            # Last file is at failure
            final_rms = bearing_data[rms_column].iloc[-1]
            final_rms_values.append(final_rms)

        # Set threshold at the specified percentile of failure RMS values
        self._threshold = float(np.percentile(final_rms_values, self.threshold_percentile))
        self._fitted = True
        return self

    def predict_bearing(
        self,
        bearing_features: pd.DataFrame,
        rms_column: str = "h_rms",
        max_rul: float = 125.0,
    ) -> TrendingPrediction:
        """Predict RUL for a single bearing's lifecycle.

        Args:
            bearing_features: DataFrame with features for one bearing, sorted by time.
            rms_column: Column name for RMS values.
            max_rul: Maximum RUL value (caps early predictions).

        Returns:
            TrendingPrediction with predicted and true RUL values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction. Call fit() first.")

        rms_values = bearing_features[rms_column].values
        n_samples = len(rms_values)
        time_indices = np.arange(n_samples)

        # Smooth RMS values
        if self.smoothing_window > 1:
            rms_smoothed = self._moving_average(rms_values, self.smoothing_window)
        else:
            rms_smoothed = rms_values

        # Predict RUL for each time point
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            if i < 2:
                # Not enough data points for trend fitting
                y_pred[i] = max_rul
                continue

            # Use data up to current point
            t = time_indices[: i + 1]
            rms = rms_smoothed[: i + 1]

            # Fit trend and extrapolate
            rul = self._estimate_rul_from_trend(t, rms, max_rul)
            y_pred[i] = rul

        # Get ground truth if available
        y_true = None
        if "rul" in bearing_features.columns:
            y_true = bearing_features["rul"].values

        bearing_id = bearing_features["bearing_id"].iloc[0]

        return TrendingPrediction(
            y_pred=y_pred,
            y_true=y_true,
            bearing_id=bearing_id,
            method=f"rms_threshold_{self.trend_type}",
            health_indicator=rms_values,
            threshold=self._threshold,
        )

    def _estimate_rul_from_trend(
        self,
        time_indices: np.ndarray,
        rms_values: np.ndarray,
        max_rul: float,
    ) -> float:
        """Estimate RUL by extrapolating trend to threshold.

        Args:
            time_indices: Time indices (0, 1, 2, ...).
            rms_values: RMS values at each time.
            max_rul: Maximum RUL value to return.

        Returns:
            Estimated RUL (files remaining until threshold crossing).
        """
        current_time = time_indices[-1]
        current_rms = rms_values[-1]

        # If already above threshold, RUL is 0
        if current_rms >= self._threshold:
            return 0.0

        if self.trend_type == "linear":
            # Linear fit: rms = a * t + b
            slope, intercept, _, _, _ = stats.linregress(time_indices, rms_values)

            if slope <= 0:
                # Not degrading, return max RUL
                return max_rul

            # Time to reach threshold: threshold = slope * t_fail + intercept
            t_fail = (self._threshold - intercept) / slope
            rul = t_fail - current_time

        else:  # exponential
            # Exponential fit: rms = a * exp(b * t)
            # Take log: log(rms) = log(a) + b * t
            log_rms = np.log(np.maximum(rms_values, 1e-10))
            slope, intercept, _, _, _ = stats.linregress(time_indices, log_rms)

            if slope <= 0:
                return max_rul

            # Time to reach threshold: log(threshold) = log(a) + b * t_fail
            log_threshold = np.log(self._threshold)
            t_fail = (log_threshold - intercept) / slope
            rul = t_fail - current_time

        # Clamp to valid range
        return float(np.clip(rul, 0.0, max_rul))

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing.

        Args:
            values: Input array.
            window: Window size.

        Returns:
            Smoothed array (same length as input).
        """
        kernel = np.ones(window) / window
        # Use same padding to maintain array length
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @property
    def threshold(self) -> float | None:
        """Get the learned failure threshold."""
        return self._threshold


class KurtosisTrendingBaseline:
    """RUL estimation based on kurtosis trending with linear fit.

    Kurtosis measures the "tailedness" of a distribution. In vibration signals,
    increased kurtosis indicates impulsive events typical of bearing faults.

    This baseline fits a linear trend to kurtosis values and estimates RUL
    based on the degradation rate.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        use_rate_based: bool = True,
    ) -> None:
        """Initialize the kurtosis trending baseline.

        Args:
            smoothing_window: Window size for moving average smoothing.
            use_rate_based: If True, estimate RUL from degradation rate.
                If False, use threshold-based approach like RMS.
        """
        self.smoothing_window = smoothing_window
        self.use_rate_based = use_rate_based
        self._healthy_kurtosis: float | None = None
        self._failure_kurtosis: float | None = None
        self._fitted = False

    def fit(
        self,
        features_df: pd.DataFrame,
        kurtosis_column: str = "h_kurtosis",
    ) -> "KurtosisTrendingBaseline":
        """Fit baseline parameters from training data.

        Args:
            features_df: DataFrame with features and metadata.
            kurtosis_column: Column name for kurtosis values.

        Returns:
            Self for method chaining.
        """
        healthy_values = []
        failure_values = []

        for bearing_id in features_df["bearing_id"].unique():
            bearing_data = features_df[features_df["bearing_id"] == bearing_id]
            n = len(bearing_data)

            # Early phase (first 10%) as healthy
            healthy_end = max(1, int(0.1 * n))
            healthy_values.extend(bearing_data[kurtosis_column].iloc[:healthy_end].tolist())

            # Final phase (last 10%) as failure
            failure_start = max(healthy_end, int(0.9 * n))
            failure_values.extend(bearing_data[kurtosis_column].iloc[failure_start:].tolist())

        self._healthy_kurtosis = float(np.median(healthy_values))
        self._failure_kurtosis = float(np.percentile(failure_values, 75))
        self._fitted = True
        return self

    def predict_bearing(
        self,
        bearing_features: pd.DataFrame,
        kurtosis_column: str = "h_kurtosis",
        max_rul: float = 125.0,
    ) -> TrendingPrediction:
        """Predict RUL for a single bearing's lifecycle.

        Args:
            bearing_features: DataFrame with features for one bearing, sorted by time.
            kurtosis_column: Column name for kurtosis values.
            max_rul: Maximum RUL value (caps early predictions).

        Returns:
            TrendingPrediction with predicted and true RUL values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction. Call fit() first.")

        kurtosis_values = bearing_features[kurtosis_column].values
        n_samples = len(kurtosis_values)
        time_indices = np.arange(n_samples)

        # Smooth kurtosis values
        if self.smoothing_window > 1:
            kurtosis_smoothed = self._moving_average(kurtosis_values, self.smoothing_window)
        else:
            kurtosis_smoothed = kurtosis_values

        # Predict RUL for each time point
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            if i < 2:
                y_pred[i] = max_rul
                continue

            t = time_indices[: i + 1]
            kurt = kurtosis_smoothed[: i + 1]

            if self.use_rate_based:
                rul = self._estimate_rul_rate_based(t, kurt, max_rul)
            else:
                rul = self._estimate_rul_threshold_based(t, kurt, max_rul)

            y_pred[i] = rul

        # Get ground truth if available
        y_true = None
        if "rul" in bearing_features.columns:
            y_true = bearing_features["rul"].values

        bearing_id = bearing_features["bearing_id"].iloc[0]

        return TrendingPrediction(
            y_pred=y_pred,
            y_true=y_true,
            bearing_id=bearing_id,
            method="kurtosis_trending",
            health_indicator=kurtosis_values,
            threshold=self._failure_kurtosis,
        )

    def _estimate_rul_rate_based(
        self,
        time_indices: np.ndarray,
        kurtosis_values: np.ndarray,
        max_rul: float,
    ) -> float:
        """Estimate RUL from degradation rate.

        Uses the rate of kurtosis increase to estimate remaining life.

        Args:
            time_indices: Time indices.
            kurtosis_values: Kurtosis values at each time.
            max_rul: Maximum RUL value.

        Returns:
            Estimated RUL.
        """
        # Fit linear trend
        slope, _, _, _, _ = stats.linregress(time_indices, kurtosis_values)

        if slope <= 0:
            # Not degrading
            return max_rul

        current_kurt = kurtosis_values[-1]
        current_time = time_indices[-1]

        # How much kurtosis increase remains before failure?
        remaining_increase = self._failure_kurtosis - current_kurt

        if remaining_increase <= 0:
            # Already at or past failure level
            return 0.0

        # Time to reach failure level at current degradation rate
        rul = remaining_increase / slope

        return float(np.clip(rul, 0.0, max_rul))

    def _estimate_rul_threshold_based(
        self,
        time_indices: np.ndarray,
        kurtosis_values: np.ndarray,
        max_rul: float,
    ) -> float:
        """Estimate RUL by extrapolating to threshold.

        Args:
            time_indices: Time indices.
            kurtosis_values: Kurtosis values at each time.
            max_rul: Maximum RUL value.

        Returns:
            Estimated RUL.
        """
        current_time = time_indices[-1]
        current_kurt = kurtosis_values[-1]

        if current_kurt >= self._failure_kurtosis:
            return 0.0

        slope, intercept, _, _, _ = stats.linregress(time_indices, kurtosis_values)

        if slope <= 0:
            return max_rul

        t_fail = (self._failure_kurtosis - intercept) / slope
        rul = t_fail - current_time

        return float(np.clip(rul, 0.0, max_rul))

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")


class HealthIndicatorFusion:
    """RUL estimation by fusing multiple health indicators.

    This baseline combines multiple health indicators (RMS, kurtosis, etc.)
    using weighted averaging to produce a more robust RUL estimate.

    Fusion approaches:
    - Simple average: Equal weights for all indicators
    - Weighted average: Learn optimal weights from training data
    - Adaptive: Adjust weights based on degradation stage
    """

    def __init__(
        self,
        indicators: list[str] | None = None,
        fusion_method: Literal["average", "weighted", "adaptive"] = "weighted",
        smoothing_window: int = 5,
    ) -> None:
        """Initialize health indicator fusion baseline.

        Args:
            indicators: List of indicator column names. Default uses RMS and kurtosis
                for both channels.
            fusion_method: Method for combining indicators.
            smoothing_window: Window size for smoothing.
        """
        self.indicators = indicators or [
            "h_rms",
            "v_rms",
            "h_kurtosis",
            "v_kurtosis",
        ]
        self.fusion_method = fusion_method
        self.smoothing_window = smoothing_window
        self._weights: np.ndarray | None = None
        self._indicator_ranges: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def fit(self, features_df: pd.DataFrame) -> "HealthIndicatorFusion":
        """Fit fusion parameters from training data.

        Args:
            features_df: DataFrame with features and metadata.

        Returns:
            Self for method chaining.
        """
        n_indicators = len(self.indicators)

        # Compute indicator ranges for normalization
        for indicator in self.indicators:
            values = features_df[indicator].values
            self._indicator_ranges[indicator] = (float(np.min(values)), float(np.max(values)))

        # Learn weights based on correlation with RUL
        if self.fusion_method == "weighted" and "rul" in features_df.columns:
            weights = []
            for indicator in self.indicators:
                # Compute correlation with RUL
                corr = np.corrcoef(features_df[indicator].values, features_df["rul"].values)[0, 1]
                # Use absolute correlation as weight
                weights.append(np.abs(corr))

            weights = np.array(weights)
            # Normalize weights to sum to 1
            self._weights = weights / np.sum(weights)
        else:
            # Equal weights
            self._weights = np.ones(n_indicators) / n_indicators

        self._fitted = True
        return self

    def predict_bearing(
        self,
        bearing_features: pd.DataFrame,
        max_rul: float = 125.0,
    ) -> TrendingPrediction:
        """Predict RUL for a single bearing's lifecycle.

        Args:
            bearing_features: DataFrame with features for one bearing, sorted by time.
            max_rul: Maximum RUL value.

        Returns:
            TrendingPrediction with predicted and true RUL values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction. Call fit() first.")

        n_samples = len(bearing_features)
        time_indices = np.arange(n_samples)

        # Normalize and combine indicators into fused health indicator
        fused_hi = self._compute_fused_hi(bearing_features)

        # Smooth the fused HI
        if self.smoothing_window > 1:
            fused_hi = self._moving_average(fused_hi, self.smoothing_window)

        # Predict RUL using linear trend extrapolation
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            if i < 2:
                y_pred[i] = max_rul
                continue

            t = time_indices[: i + 1]
            hi = fused_hi[: i + 1]

            rul = self._estimate_rul_from_hi(t, hi, max_rul)
            y_pred[i] = rul

        # Get ground truth if available
        y_true = None
        if "rul" in bearing_features.columns:
            y_true = bearing_features["rul"].values

        bearing_id = bearing_features["bearing_id"].iloc[0]

        return TrendingPrediction(
            y_pred=y_pred,
            y_true=y_true,
            bearing_id=bearing_id,
            method=f"fusion_{self.fusion_method}",
            health_indicator=fused_hi,
            threshold=1.0,  # Normalized HI reaches 1.0 at failure
        )

    def _compute_fused_hi(self, bearing_features: pd.DataFrame) -> np.ndarray:
        """Compute fused health indicator from multiple indicators.

        Args:
            bearing_features: DataFrame with indicator columns.

        Returns:
            Fused health indicator array (0=healthy, 1=failed).
        """
        n_samples = len(bearing_features)
        fused = np.zeros(n_samples)

        for i, indicator in enumerate(self.indicators):
            values = bearing_features[indicator].values
            vmin, vmax = self._indicator_ranges[indicator]

            # Normalize to [0, 1] range
            if vmax > vmin:
                normalized = (values - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(values)

            # Clip to [0, 1]
            normalized = np.clip(normalized, 0.0, 1.0)

            # Apply weight
            fused += self._weights[i] * normalized

        return fused

    def _estimate_rul_from_hi(
        self,
        time_indices: np.ndarray,
        hi_values: np.ndarray,
        max_rul: float,
    ) -> float:
        """Estimate RUL from fused health indicator.

        Args:
            time_indices: Time indices.
            hi_values: Fused health indicator values (0=healthy, 1=failed).
            max_rul: Maximum RUL value.

        Returns:
            Estimated RUL.
        """
        current_time = time_indices[-1]
        current_hi = hi_values[-1]

        # If HI >= 1.0, bearing has failed
        if current_hi >= 1.0:
            return 0.0

        # Fit linear trend
        slope, intercept, _, _, _ = stats.linregress(time_indices, hi_values)

        if slope <= 0:
            return max_rul

        # Extrapolate to HI = 1.0
        t_fail = (1.0 - intercept) / slope
        rul = t_fail - current_time

        return float(np.clip(rul, 0.0, max_rul))

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @property
    def weights(self) -> np.ndarray | None:
        """Get the learned fusion weights."""
        return self._weights

    @property
    def indicator_weights_dict(self) -> dict[str, float]:
        """Get weights as a dictionary mapping indicator name to weight."""
        if self._weights is None:
            return {}
        return dict(zip(self.indicators, self._weights.tolist()))


def predict_all_bearings(
    model: RMSThresholdBaseline | KurtosisTrendingBaseline | HealthIndicatorFusion,
    features_df: pd.DataFrame,
    **predict_kwargs,
) -> list[TrendingPrediction]:
    """Run prediction for all bearings in the dataset.

    Args:
        model: Fitted trending baseline model.
        features_df: DataFrame with features for all bearings.
        **predict_kwargs: Additional arguments passed to predict_bearing().

    Returns:
        List of TrendingPrediction objects, one per bearing.
    """
    predictions = []

    for bearing_id in features_df["bearing_id"].unique():
        bearing_data = features_df[features_df["bearing_id"] == bearing_id].sort_values("file_idx")
        pred = model.predict_bearing(bearing_data, **predict_kwargs)
        predictions.append(pred)

    return predictions


def evaluate_trending_baseline(
    predictions: list[TrendingPrediction],
) -> dict[str, float]:
    """Evaluate trending baseline predictions.

    Args:
        predictions: List of TrendingPrediction objects.

    Returns:
        Dictionary with aggregate metrics (RMSE, MAE, etc.).
    """
    all_y_true = []
    all_y_pred = []

    for pred in predictions:
        if pred.y_true is not None:
            all_y_true.extend(pred.y_true.tolist())
            all_y_pred.extend(pred.y_pred.tolist())

    if not all_y_true:
        return {}

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Import metrics from training module
    from src.training.metrics import evaluate_predictions

    return evaluate_predictions(y_true, y_pred)
