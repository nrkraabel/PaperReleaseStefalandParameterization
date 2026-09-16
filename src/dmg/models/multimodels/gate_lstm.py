"""LSTM-based gating network for Mixture of Experts ensembles.

Uses a standard LSTM over temporal gate forcings (expert predictions,
meteorological features, etc.) to produce per-timestep gating weights.
Static catchment attributes are concatenated with the temporal input at
each timestep.
"""

from typing import Optional

import torch
import torch.nn as nn


class GateLSTM(nn.Module):
    """LSTM-based gating network for Mixture of Experts.

    Produces per-timestep logits over *K* experts using an LSTM over
    temporal forcings, optionally conditioned on static catchment
    attributes (concatenated at each timestep).

    Parameters
    ----------
    n_input
        Number of temporal input features.
    n_experts
        Number of experts (output dimension).
    n_attributes
        Number of static attributes to concatenate. Set to 0 to
        disable attribute conditioning.
    hidden_size
        LSTM hidden dimension.
    num_layers
        Number of stacked LSTM layers.
    dropout
        Dropout rate applied between LSTM layers and in the output
        head.
    """

    def __init__(
        self,
        n_input: int,
        n_experts: int,
        n_attributes: int = 0,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        total_input = n_input + n_attributes

        self.lstm = nn.LSTM(
            input_size=total_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, n_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Temporal input ``[B, T, n_input]``.
        a
            Static attributes ``[B, n_attributes]`` or ``None``.

        Returns
        -------
        torch.Tensor
            Raw logits ``[B, T, n_experts]`` (pre-softmax).
        """
        if a is not None:
            a_tiled = a.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, a_tiled], dim=-1)

        h, _ = self.lstm(x)  # [B, T, hidden_size]
        return self.head(h)  # [B, T, n_experts]
