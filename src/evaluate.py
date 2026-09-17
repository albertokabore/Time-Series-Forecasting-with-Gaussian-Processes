"""Evaluation metrics for probabilistic GP forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float
    mean_nlpd: float  # mean negative log predictive density (nats), per point
    coverage_95: float  # fraction of observations inside the 95% predictive interval

    def as_dict(self) -> dict:
        return {
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "Mean predictive log likelihood (nats)": -self.mean_nlpd,
            "Mean NLPD": self.mean_nlpd,
            "95% predictive interval coverage": self.coverage_95,
        }


def compute_metrics(
    y_true: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
) -> Metrics:
    """Compute point-forecast and probabilistic metrics on the ORIGINAL scale.

    ``pred_mean``/``pred_std`` are the predictive mean and standard deviation
    of the observation (i.e. including likelihood noise), already unscaled
    to the original units of the target.
    """

    y = y_true.numpy()
    mu = pred_mean.numpy()
    sigma = pred_std.numpy()

    errors = y - mu
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    # Negative log predictive density under N(mu, sigma^2) per point.
    nlpd = 0.5 * np.log(2 * np.pi * sigma**2) + (errors**2) / (2 * sigma**2)
    mean_nlpd = float(np.mean(nlpd))

    lower = mu - 1.96 * sigma
    upper = mu + 1.96 * sigma
    coverage = float(np.mean((y >= lower) & (y <= upper)))

    return Metrics(mse=mse, rmse=rmse, mae=mae, mean_nlpd=mean_nlpd, coverage_95=coverage)
