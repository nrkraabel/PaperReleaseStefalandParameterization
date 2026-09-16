"""Automatic Weighted Loss for multi-task learning.

Based on Kendall et al. 2018 "Multi-Task Learning Using Uncertainty
to Weigh Losses".  Adapted for MoE expert specialization where
multiple auxiliary losses (oracle, diversity, load balance, WTA)
operate at different scales and should be balanced automatically.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class AutoWeightedLoss(nn.Module):
    """Automatic Weighted Loss for multi-task learning.

    Learns per-task uncertainty parameters (sigma) that control the
    relative weighting:

        L = sum_i [ 0.5 / sigma_i^2 * L_i + log(1 + sigma_i^2) ]

    The log-barrier prevents any sigma from going to infinity (which
    would zero out that loss).  Higher uncertainty -> lower weight.

    Parameters
    ----------
    num_losses
        Number of loss terms to weight.
    init_sigmas
        Initial sigma values.  Defaults to all ones.
    clamp_range
        Min/max for sigma to prevent instability.  The minimum sigma
        controls max amplification: 0.5/sigma_min^2.  Default (0.3, 10)
        gives max amplification ~5.5x.
    """

    def __init__(
        self,
        num_losses: int,
        init_sigmas: Optional[list[float]] = None,
        clamp_range: tuple[float, float] = (0.3, 10.0),
    ):
        super().__init__()
        if init_sigmas is None:
            params = torch.ones(num_losses)
        else:
            params = torch.tensor(init_sigmas, dtype=torch.float32)
        self.params = nn.Parameter(params)
        self.clamp_range = clamp_range
        self.num_losses = num_losses

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        """Combine losses with learned weights.

        Parameters
        ----------
        *losses
            One scalar loss per task (must match ``num_losses``).
            NaN/Inf losses are replaced with zero (skipped).

        Returns
        -------
        torch.Tensor
            Combined weighted loss.
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")
        sigma = torch.clamp(self.params, *self.clamp_range)
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            # Skip NaN/Inf losses to prevent poison propagation.
            if not torch.isfinite(loss):
                continue
            s2 = sigma[i] ** 2
            total = total + 0.5 / s2 * loss + torch.log(1.0 + s2)
        return total

    def get_weights(self) -> np.ndarray:
        """Return normalized effective weights ``1/sigma^2``."""
        with torch.no_grad():
            sigma = torch.clamp(self.params, *self.clamp_range)
            w = 1.0 / (sigma**2)
            return (w / w.sum()).cpu().numpy()

    def get_sigmas(self) -> np.ndarray:
        """Return current sigma values."""
        with torch.no_grad():
            return torch.clamp(self.params, *self.clamp_range).cpu().numpy()
