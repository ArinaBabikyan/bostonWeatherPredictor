"""
Preprocessing & windowing utilities for weather forecasting.
Fill in FEATURE_NAMES to match your training DataFrame columns order.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List

HISTORY_WINDOW = 72
FORECAST_HORIZON = 24
TARGET_COLUMN_IDX = 0  # adjust if needed

# IMPORTANT: set this to the exact order you trained with
FEATURE_NAMES: List[str] = [
    # example: replace with your actual columns in df_hourly
    # 'temperature', 'relative_humidity', 'pressure', 'wind_speed', 'wind_dir',
    # 'cloud_cover', 'precipitation', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy'
    'temperature'
]


def create_windows(data: np.ndarray,
                   target_column_idx: int,
                   history_window: int = HISTORY_WINDOW,
                   forecast_horizon: int = FORECAST_HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a (T, F) array, create samples of shape (N, history_window, F) and targets (N,).
    """
    X, y = [], []
    max_start = len(data) - history_window - forecast_horizon
    for start_idx in range(max_start):
        end_idx = start_idx + history_window
        target_idx = end_idx + forecast_horizon - 1
        X.append(data[start_idx:end_idx, :])
        y.append(data[target_idx, target_column_idx])
    return np.array(X), np.array(y)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time-of-day and day-of-year features to a DateTime-indexed DF."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex for time features.")
    df = df.copy()
    df['hour'] = df.index.hour
    df['doy'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_doy']  = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_doy']  = np.cos(2 * np.pi * df['doy'] / 365.25)
    return df


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DF columns are in the exact order used for training."""
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[FEATURE_NAMES]
