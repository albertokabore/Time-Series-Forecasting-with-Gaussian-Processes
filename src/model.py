"""Gaussian Process model for the Delhi daily temperature series.

Kernel design
-------------
Daily mean temperature is dominated by an annual seasonal cycle plus a
slower-moving trend/local drift, so we use a sum of three components,
following the classic structure recommended for climate/seasonal series
(cf. the GPML book's treatment of the Mauna Loa CO2 series):

  * ``RBFKernel``            - smooth long-term trend
  * ``PeriodicKernel``       - the ~365-day annual seasonal cycle
  * ``RBFKernel`` (shorter)  - short-range day-to-day correlation

Each component has its own ``ScaleKernel`` output-scale, and the
``GaussianLikelihood`` supplies the observation noise term. All lengthscales
and the periodic kernel's period are free parameters, optimized jointly by
maximizing the exact marginal log likelihood.
"""

from __future__ import annotations

import gpytorch
import torch


class DelhiClimateGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # Constrain lengthscales so `trend` and `local` keep their intended,
        # interpretable roles (long-range drift vs. short-range/day-to-day
        # correlation) instead of drifting into each other's regime during
        # optimization.
        trend = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.GreaterThan(0.5)
            )
        )
        local = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.LessThan(0.15)
            )
        )
        seasonal = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                period_length_constraint=gpytorch.constraints.Interval(0.8, 1.2)
            )
        )

        trend.base_kernel.lengthscale = torch.tensor(2.0)
        local.base_kernel.lengthscale = torch.tensor(0.1)
        # Initialize the period near 1.0 (= 365 days, since x is in years).
        seasonal.base_kernel.period_length = torch.tensor(1.0)

        self.covar_module = trend + seasonal + local

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_model(
    train_x: torch.Tensor, train_y: torch.Tensor
) -> tuple[DelhiClimateGP, gpytorch.likelihoods.GaussianLikelihood]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DelhiClimateGP(train_x, train_y, likelihood)
    return model, likelihood
