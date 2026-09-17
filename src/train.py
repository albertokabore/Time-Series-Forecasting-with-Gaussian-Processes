"""Training loop for the Delhi climate GP model."""

from __future__ import annotations

import gpytorch
import torch

from src.model import DelhiClimateGP


def train_model(
    model: DelhiClimateGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int = 500,
    lr: float = 0.02,
) -> list[float]:
    """Optimize kernel hyperparameters and noise by maximizing the exact MLL."""

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history: list[float] = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if (i + 1) % 25 == 0 or i == 0:
            print(f"Iter {i + 1:>4}/{n_iter} - loss: {loss.item():.4f} "
                  f"- noise: {likelihood.noise.item():.4f}")

    return loss_history
