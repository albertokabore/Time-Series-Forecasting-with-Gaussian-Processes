"""Plotting helpers for the GP forecast report."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import Dataset


def plot_full_forecast(
    ds: Dataset,
    test_mean: torch.Tensor,
    test_std: torch.Tensor,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(ds.train_dates, ds.unscale_y(ds.y_train).numpy(), color="#1f77b4",
            label="Train (observed)", linewidth=0.8)
    ax.plot(ds.test_dates, ds.unscale_y(ds.y_test).numpy(), color="#2ca02c",
            label="Test (observed)", linewidth=1.2)

    mean = test_mean.numpy()
    std = test_std.numpy()
    ax.plot(ds.test_dates, mean, color="#d62728", label="GP forecast (mean)", linewidth=1.2)
    ax.fill_between(
        ds.test_dates,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="#d62728",
        alpha=0.2,
        label="95% predictive interval",
    )

    ax.set_title("Delhi Daily Mean Temperature: GP Forecast vs. Observed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_zoom(
    ds: Dataset,
    test_mean: torch.Tensor,
    test_std: torch.Tensor,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    mean = test_mean.numpy()
    std = test_std.numpy()
    true = ds.unscale_y(ds.y_test).numpy()

    ax.plot(ds.test_dates, true, "o-", color="#2ca02c", label="Observed", markersize=3)
    ax.plot(ds.test_dates, mean, color="#d62728", label="GP forecast (mean)")
    ax.fill_between(
        ds.test_dates,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="#d62728",
        alpha=0.25,
        label="95% predictive interval",
    )
    ax.set_title("Test-Period Forecast (Zoomed In)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss(loss_history: list[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, color="#1f77b4")
    ax.set_title("Training Loss (Negative Marginal Log Likelihood)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negative MLL")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    ds: Dataset,
    test_mean: torch.Tensor,
    out_path: str,
) -> None:
    true = ds.unscale_y(ds.y_test).numpy()
    residuals = true - test_mean.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ds.test_dates, residuals, color="#9467bd")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Residuals over Test Period")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual (°C)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].hist(residuals, bins=20, color="#9467bd", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (°C)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
