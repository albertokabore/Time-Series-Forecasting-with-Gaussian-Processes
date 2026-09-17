"""Data loading and preprocessing for the Delhi daily climate time series.

The raw files (``DailyDelhiClimateTrain.csv`` / ``DailyDelhiClimateTest.csv``)
already form a chronological train/test split, so preprocessing here is
limited to what the raw data actually needs: parsing dates, deduplicating the
one overlapping date between the two files, converting dates to a numeric time axis, and
standardizing the target for numerically stable GP optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

TARGET_COLUMN = "meantemp"

@dataclass
class Dataset:
    """Container for a preprocessed train/test split on a single target."""

    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    x_train: torch.Tensor  # standardized time index, shape (n_train,)
    y_train: torch.Tensor  # standardized target, shape (n_train,)
    x_test: torch.Tensor
    y_test: torch.Tensor
    y_mean: float
    y_std: float
    x_scale: float  # days -> scaled-time divisor used for x

    def unscale_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


def _load_raw(train_path: str, test_path: str) -> pd.DataFrame:
    train = pd.read_csv(train_path, parse_dates=["date"])
    test = pd.read_csv(test_path, parse_dates=["date"])

    # The two files share 2017-01-01. Keep the training copy and drop the
    # duplicate from test so no observation is double-counted.
    overlap = set(train["date"]) & set(test["date"])
    if overlap:
        test = test[~test["date"].isin(overlap)].reset_index(drop=True)

    for df in (train, test):
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)

    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)
    return train, test


def load_dataset(
    train_path: str = "DailyDelhiClimateTrain.csv",
    test_path: str = "DailyDelhiClimateTest.csv",
    target: str = TARGET_COLUMN,
) -> Dataset:
    """Load, clean, and standardize the climate series into GP-ready tensors."""

    train_df, test_df = _load_raw(train_path, test_path)

    origin = train_df["date"].iloc[0]
    x_scale = 365.0  # express time in years since the start of the series

    def to_x(df: pd.DataFrame) -> np.ndarray:
        return ((df["date"] - origin).dt.days.to_numpy(dtype=np.float64)) / x_scale

    x_train_np = to_x(train_df)
    x_test_np = to_x(test_df)

    y_train_np = train_df[target].to_numpy(dtype=np.float64)
    y_test_np = test_df[target].to_numpy(dtype=np.float64)

    y_mean = float(y_train_np.mean())
    y_std = float(y_train_np.std())

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_train = torch.tensor((y_train_np - y_mean) / y_std, dtype=torch.float32)
    y_test = torch.tensor((y_test_np - y_mean) / y_std, dtype=torch.float32)

    return Dataset(
        train_dates=pd.DatetimeIndex(train_df["date"]),
        test_dates=pd.DatetimeIndex(test_df["date"]),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_mean=y_mean,
        y_std=y_std,
        x_scale=x_scale,
    )
