"""TCN-based gating network for Mixture of Experts ensembles.

Adapted from the EnhancedGatedTCN surrogate architecture. Uses causal
dilated convolutions over expert predictions to produce temporally-aware
gating weights.  Static catchment attributes are injected via Dynamic
FiLM conditioning.
"""

from typing import Optional

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Recalibrates channel-wise feature responses using global (or causal
    cumulative) pooling and a learnable bottleneck MLP.

    Parameters
    ----------
    channels
        Number of input channels.
    reduction
        Reduction ratio for the bottleneck.
    causal
        If ``True``, uses cumulative statistics so position *t* only
        sees information from timesteps 1 ... t.
    """

    def __init__(self, channels: int, reduction: int = 4, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor ``[B, C, T]``.

        Returns
        -------
        torch.Tensor
            Reweighted tensor ``[B, C, T]``.
        """
        if self.causal:
            B, C, T = x.shape
            cumsum = torch.cumsum(x, dim=2)
            positions = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
            scale = cumsum / positions.view(1, 1, T)
        else:
            scale = self.squeeze(x)

        scale = self.excitation(scale)
        return x * scale


class DynamicFiLM(nn.Module):
    """Dynamic Feature-wise Linear Modulation.

    Generates both static and temporal-adaptive modulation parameters
    from conditioning features (e.g. catchment attributes).

    Parameters
    ----------
    a_dim
        Dimension of conditioning features.
    c_dim
        Number of channels to modulate.
    hidden
        Hidden dimension for modulation networks.
    """

    def __init__(self, a_dim: int, c_dim: int, hidden: int = 128):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(a_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * c_dim),
        )
        self.dynamic_net = nn.Sequential(
            nn.Linear(a_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.temporal_conv = nn.Conv1d(hidden, 2 * c_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor ``[B, C, T]``.
        A
            Conditioning features ``[B, a_dim]``.

        Returns
        -------
        torch.Tensor
            Modulated tensor ``[B, C, T]``.
        """
        B, C, T = x.shape

        gamma_s, beta_s = self.static_net(A).chunk(2, dim=-1)
        gamma_s = gamma_s.unsqueeze(-1)
        beta_s = beta_s.unsqueeze(-1)

        h = self.dynamic_net(A).unsqueeze(-1)
        h = h.expand(-1, -1, T)
        gamma_d, beta_d = self.temporal_conv(h).chunk(2, dim=1)

        gamma = (gamma_s + gamma_d) * 0.5
        beta = (beta_s + beta_d) * 0.5

        return gamma * x + beta


class MultiScaleGatedTCNBlock(nn.Module):
    """Multi-scale gated TCN block with causal convolutions.

    Features: multi-timescale dilations, multi-head GLU gating, SE
    attention, Dynamic FiLM conditioning, pre/post LayerNorm,
    stochastic depth, and learnable skip connections.

    Parameters
    ----------
    channels
        Number of channels.
    kernel_size
        Convolution kernel size.
    dilations
        List of dilation rates for parallel branches.
    dropout
        Dropout rate.
    a_dim
        Dimension of conditioning features (``None`` to disable FiLM).
    stochastic_depth
        Probability of dropping the block during training.
    use_se
        Whether to use SE attention.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Optional[list[int]] = None,
        dropout: float = 0.0,
        a_dim: Optional[int] = None,
        stochastic_depth: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        self.stochastic_depth = stochastic_depth

        if dilations is None:
            dilations = [1]
        self.dilations = dilations

        # Multi-branch causal convolutions.
        self.conv_branches = nn.ModuleList()
        branch_channels = (2 * channels) // len(dilations)

        for dilation in dilations:
            pad = (kernel_size - 1) * dilation  # causal: all padding on left
            self.conv_branches.append(
                nn.Conv1d(
                    channels,
                    branch_channels,
                    kernel_size,
                    padding=pad,
                    dilation=dilation,
                )
            )

        self.channel_adjustment = None
        total_branch_channels = branch_channels * len(dilations)
        if total_branch_channels != 2 * channels:
            self.channel_adjustment = nn.Conv1d(
                total_branch_channels,
                2 * channels,
                1,
            )

        self.film = DynamicFiLM(a_dim, channels) if a_dim is not None else None
        self.se = SEBlock(channels, causal=True) if use_se else nn.Identity()

        self.pre_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.post_norm = nn.LayerNorm(channels)

        self.skip_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor ``[B, C, T]``.
        A
            Conditioning features ``[B, a_dim]`` or ``None``.

        Returns
        -------
        torch.Tensor
            Output tensor ``[B, C, T]``.
        """
        res = x

        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return res

        # Pre-activation normalization.
        h = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)

        # Multi-scale causal convolutions.
        branch_outputs = []
        for conv in self.conv_branches:
            branch_h = conv(h)
            trim = (conv.kernel_size[0] - 1) * conv.dilation[0]
            if trim > 0:
                branch_h = branch_h[..., :-trim]
            branch_outputs.append(branch_h)

        h = torch.cat(branch_outputs, dim=1)

        if self.channel_adjustment is not None:
            h = self.channel_adjustment(h)

        # Multi-head GLU gating.
        num_heads = 2
        head_dim = h.size(1) // (2 * num_heads)

        gated_heads = []
        for i in range(num_heads):
            start_idx = i * 2 * head_dim
            h_in = h[:, start_idx : start_idx + head_dim, :]
            h_gate = h[:, start_idx + head_dim : start_idx + 2 * head_dim, :]
            gated_heads.append(torch.tanh(h_in) * torch.sigmoid(h_gate))

        h = torch.cat(gated_heads, dim=1)

        # FiLM conditioning.
        if self.film is not None and A is not None:
            h = self.film(h, A)

        # SE attention.
        h = self.se(h)

        # Dropout + post-normalization.
        h = self.dropout(h)
        h = self.post_norm(h.transpose(1, 2)).transpose(1, 2)

        return self.skip_weight * h + res


class GateTCN(nn.Module):
    """TCN-based gating network for Mixture of Experts.

    Produces per-timestep logits over *K* experts using causal dilated
    convolutions over expert predictions, optionally conditioned on
    static catchment attributes via Dynamic FiLM.

    Parameters
    ----------
    n_input
        Number of temporal input features (depends on ``input_mode``:
        *K* for ``raw_q``, *K*(K-1)/2* for ``residual``, etc.).
    n_experts
        Number of experts (output dimension).
    n_attributes
        Number of static attributes for FiLM conditioning. Set to 0
        to disable FiLM.
    width
        Channel width of TCN blocks.
    depth
        Number of TCN blocks.
    kernel_size
        Convolution kernel size.
    dropout
        Dropout rate.
    stochastic_depth
        Maximum stochastic depth probability (linearly scaled per
        block).
    use_se
        Whether to use Squeeze-and-Excitation attention.
    multi_scale
        Whether to use multi-scale dilations in early blocks.
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
        stochastic_depth: float = 0.1,
        use_se: bool = True,
        multi_scale: bool = True,
    ):
        super().__init__()

        # Input projection.
        self.inp = nn.Sequential(
            nn.Linear(n_input, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        # TCN blocks with exponential dilation growth.
        blocks = []
        for i in range(depth):
            base_dil = 2**i

            if multi_scale and i < depth // 2:
                dilations = [base_dil, base_dil * 2]
            else:
                dilations = [base_dil]

            drop_prob = stochastic_depth * (i / max(depth, 1))

            blocks.append(
                MultiScaleGatedTCNBlock(
                    channels=width,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                    a_dim=n_attributes if n_attributes > 0 else None,
                    stochastic_depth=drop_prob,
                    use_se=use_se,
                )
            )

        self.blocks = nn.ModuleList(blocks)

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
            Temporal input ``[B, T, n_input]`` (e.g. expert
            predictions stacked along the feature dimension).
        a
            Static attributes ``[B, n_attributes]`` for FiLM
            conditioning, or ``None``.

        Returns
        -------
        torch.Tensor
            Raw logits ``[B, T, n_experts]`` (pre-softmax).
        """
        h = self.inp(x)  # [B, T, width]
        h = h.transpose(1, 2)  # [B, width, T]

        for blk in self.blocks:
            h = blk(h, A=a)

        y = self.head(h)  # [B, n_experts, T]
        return y.transpose(1, 2)  # [B, T, n_experts]
