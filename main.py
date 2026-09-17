"""End-to-end pipeline: load data, train a GP, forecast, evaluate, plot.

Run with:  python main.py
Outputs land in outputs/figures and outputs/metrics/metrics.json
"""

from __future__ import annotations

import json
import os

import torch

from src.data import load_dataset
from src.evaluate import compute_metrics
from src.model import build_model
from src.plotting import plot_full_forecast, plot_loss, plot_residuals, plot_test_zoom
from src.train import train_model

SEED = 42


def main() -> None:
    torch.manual_seed(SEED)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    print("Loading and preprocessing data...")
    ds = load_dataset()
    print(f"  Train: {len(ds.x_train)} days ({ds.train_dates[0].date()} to {ds.train_dates[-1].date()})")
    print(f"  Test:  {len(ds.x_test)} days ({ds.test_dates[0].date()} to {ds.test_dates[-1].date()})")

    print("\nBuilding model...")
    model, likelihood = build_model(ds.x_train, ds.y_train)

    print("\nTraining...")
    loss_history = train_model(model, likelihood, ds.x_train, ds.y_train, n_iter=500, lr=0.02)

    print("\nLearned hyperparameters:")
    trend, seasonal, local = model.covar_module.kernels
    hyperparams = {
        "trend_lengthscale_years": trend.base_kernel.lengthscale.item(),
        "seasonal_period_years": seasonal.base_kernel.period_length.item(),
        "seasonal_lengthscale": seasonal.base_kernel.lengthscale.item(),
        "local_lengthscale_days": local.base_kernel.lengthscale.item() * 365,
        "observation_noise_std_scaled": likelihood.noise.sqrt().item(),
        "trend_outputscale": trend.outputscale.item(),
        "seasonal_outputscale": seasonal.outputscale.item(),
        "local_outputscale": local.outputscale.item(),
    }
    print(f"  Trend RBF lengthscale:    {hyperparams['trend_lengthscale_years']:.3f} years")
    print(f"  Seasonal period:          {hyperparams['seasonal_period_years']:.3f} years")
    print(f"  Seasonal lengthscale:     {hyperparams['seasonal_lengthscale']:.3f}")
    print(f"  Local RBF lengthscale:    {hyperparams['local_lengthscale_days']:.1f} days")
    print(f"  Observation noise (std):  {hyperparams['observation_noise_std_scaled']:.4f} (standardized units)")
    with open("outputs/metrics/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    print("\nForecasting on test period...")
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        posterior = likelihood(model(ds.x_test))
        test_mean_std_scale = posterior.mean
        test_std_std_scale = posterior.variance.sqrt()

    test_mean = ds.unscale_y(test_mean_std_scale)
    test_std = test_std_std_scale * ds.y_std  # std scales linearly, no mean shift

    print("Evaluating...")
    y_true = ds.unscale_y(ds.y_test)
    metrics = compute_metrics(y_true, test_mean, test_std)
    print(json.dumps(metrics.as_dict(), indent=2))

    with open("outputs/metrics/metrics.json", "w") as f:
        json.dump(metrics.as_dict(), f, indent=2)

    dataset_info = {
        "n_train": len(ds.x_train),
        "n_test": len(ds.x_test),
        "train_start": str(ds.train_dates[0].date()),
        "train_end": str(ds.train_dates[-1].date()),
        "test_start": str(ds.test_dates[0].date()),
        "test_end": str(ds.test_dates[-1].date()),
        "target_mean": ds.y_mean,
        "target_std": ds.y_std,
    }
    with open("outputs/metrics/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\nSaving plots...")
    plot_full_forecast(ds, test_mean, test_std, "outputs/figures/full_forecast.png")
    plot_test_zoom(ds, test_mean, test_std, "outputs/figures/test_zoom.png")
    plot_loss(loss_history, "outputs/figures/training_loss.png")
    plot_residuals(ds, test_mean, "outputs/figures/residuals.png")

    print("Done. See outputs/figures/ and outputs/metrics/metrics.json")


if __name__ == "__main__":
    main()
