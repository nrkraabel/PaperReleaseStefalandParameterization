"""Simple causal CNN gating network for Mixture of Experts ensembles.

Uses stacked causal 1-D convolutions with residual connections over
temporal gate forcings to produce per-timestep gating weights.  Static
catchment attributes are concatenated with the temporal input at each
timestep.  Compared to the TCN gate, this is a lighter architecture
without multi-scale dilations, SE attention, or FiLM conditioning.
"""

from typing import Optional

import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    """Single causal Conv1d block with residual connection.

    Parameters
    ----------
    channels
        Number of input/output channels.
    kernel_size
        Convolution kernel size.
    dilation
        Dilation rate for the convolution.
    dropout
        Dropout rate.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor ``[B, C, T]``.

        Returns
        -------
        torch.Tensor
            Output tensor ``[B, C, T]``.
        """
        res = x
        h = self.conv(x)
        # Causal trim: remove future timesteps from right.
        if self.pad > 0:
            h = h[..., : -self.pad]
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)
        return h + res


class GateCNN(nn.Module):
    """Simple causal CNN gating network for Mixture of Experts.

    Produces per-timestep logits over *K* experts using stacked causal
    1-D convolutions over temporal forcings.  Static attributes are
    concatenated at each timestep.

    Parameters
    ----------
    n_input
        Number of temporal input features.
    n_experts
        Number of experts (output dimension).
    n_attributes
        Number of static attributes to concatenate. Set to 0 to
        disable attribute conditioning.
    width
        Channel width of convolution blocks.
    depth
        Number of convolution blocks.
    kernel_size
        Convolution kernel size.
    dropout
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_experts: int,
        n_attributes: int = 0,
        width: int = 32,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_dilation: bool = False,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        total_input = n_input + n_attributes

        # Input projection.
        self.inp = nn.Sequential(
            nn.Linear(total_input, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        # Stacked causal conv blocks.
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    width,
                    kernel_size,
                    dilation=2**i if use_dilation else 1,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

        # Output head.
        self.head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(width, n_experts, kernel_size=1),
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

        h = self.inp(x)  # [B, T, width]
        h = h.transpose(1, 2)  # [B, width, T]

        for blk in self.blocks:
            h = blk(h)

        y = self.head(h)  # [B, n_experts, T]
        return y.transpose(1, 2)  # [B, T, n_experts]
