"""Mixture of Experts (MoE) for differentiable model ensembles.

Loads pretrained DplModel experts from separate configs and checkpoints,
freezes them, and trains a gating network that produces per-basin (or
per-timestep) weights over the experts.

Gate types
----------
- ``mlp`` (default): Lightweight MLP gate; supports static and temporal
  modes based on forcings/attributes.  When ``gate.forcings`` is
  specified, operates in temporal mode using expert-derived features.
- ``tcn``: Causal temporal convolution network gate with multi-scale
  dilations, SE attention, and FiLM conditioning from static attributes.
- ``lstm``: LSTM gate over temporal forcings with static attributes
  concatenated at each timestep.
- ``cnn``: Simple causal CNN gate with residual connections and static
  attributes concatenated at each timestep.

Gate modes (MLP legacy only)
----------------------------
- ``static``: Gate sees only static attributes -> one weight per basin.
- ``temporal``: Gate sees forcings + static attributes at each timestep
  -> time-varying weights per basin.

Gate forcings
-------------
The ``gate.forcings`` list selects which temporal features are
concatenated and fed to temporal gates (TCN, LSTM, CNN, or MLP when
``gate.forcings`` is specified).  Special keywords that derive from
expert outputs:

- ``q_prime``: Unrouted streamflow (``streamflow_no_rout``) from each
  expert -> *K* features.
- ``q_routed``: Routed streamflow (``streamflow``) from each expert ->
  *K* features.
- ``q_residual``: Pairwise differences between expert routed
  predictions -> *K*(K-1)/2* features.

Any other name (e.g. ``prcp``, ``tmean``, ``pet``) is looked up by
position in ``model.nn.forcings`` and pulled from ``xc_nn_norm``.
"""

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from dmg.core.utils.utils import find_shared_keys
from dmg.models.delta_models.dpl_model import DplModel

log = logging.getLogger(__name__)


class GateMLP(nn.Module):
    """Lightweight MLP gate for Mixture of Experts.

    Produces per-basin (or per-timestep) logits over K experts.
    ``nn.Linear`` handles arbitrary leading batch dimensions, so the
    same module works for both ``[N, D]`` and ``[T, N, D]`` inputs.

    Parameters
    ----------
    n_input
        Number of input features.
    n_experts
        Number of experts to produce weights for.
    hidden_size
        Hidden layer size.
    dropout
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_experts: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input features -- ``[N, D]`` (static) or ``[T, N, D]``
            (temporal).

        Returns
        -------
        torch.Tensor
            Raw logits -- ``[N, K]`` or ``[T, N, K]``.
        """
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts combining frozen pretrained DplModels.

    Each expert is a full DplModel (NN -> physics) loaded from its own
    config YAML and checkpoint. A trainable gate (MLP or TCN) produces
    per-basin weights over the experts.

    Parameters
    ----------
    moe_config
        MoE-specific config section with keys ``experts`` (list of
        dicts) and ``gate`` (dict with type, hidden_size, dropout, etc.).
    model_config
        The main config's ``model`` section. Used to determine
        the number of forcings (for extracting attribute columns
        from ``xc_nn_norm``).
    device
        Torch device.
    """

    def __init__(
        self,
        moe_config: dict[str, Any],
        model_config: dict[str, Any],
        device: torch.device,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.moe_config = moe_config
        self.model_config = model_config
        self.device = device
        self._model_dir = model_dir
        self.scaling_function = moe_config.get('scaling_function', 'softmax')
        self.freeze_experts = moe_config.get('freeze_experts', True)
        self.temperature = moe_config.get('temperature', 1.0)
        self._gumbel_hard = moe_config.get('gumbel_hard', False)

        # Temperature annealing.
        self._temp_start = moe_config.get('temp_start', None)
        self._temp_end = moe_config.get('temp_end', None)
        self._temp_anneal_epochs = moe_config.get('temp_anneal_epochs', None)
        self._temp_anneal_schedule = moe_config.get(
            'temp_anneal_schedule',
            'linear',
        )

        # Number of forcings -- used to extract forcing columns from
        # xc_nn_norm for temporal gating.
        self.n_forcings = len(model_config['nn']['forcings'])
        self.n_attributes = len(model_config['nn']['attributes'])

        # Gate configuration.
        gate_cfg = moe_config.get('gate', {})
        self.gate_type = gate_cfg.get('type', 'mlp')
        self.gate_mode = gate_cfg.get('mode', 'static')
        self.n_gate_attrs = len(gate_cfg.get('attributes', []))

        # Temporal gate forcings -- list of temporal feature names.
        # For TCN/LSTM/CNN this is always used (default: ['q_prime']).
        # For MLP, only used when explicitly specified in config.
        if self.gate_type in ('tcn', 'lstm', 'cnn'):
            self.gate_forcings = gate_cfg.get('forcings', ['q_prime'])
            self._uses_temporal_forcings = True
        elif 'forcings' in gate_cfg:
            self.gate_forcings = gate_cfg['forcings']
            self._uses_temporal_forcings = True
        else:
            self.gate_forcings = []
            self._uses_temporal_forcings = False

        # Build a map from model.nn.forcings names to column indices in
        # xc_nn_norm so we can look up standard forcings by name.
        self._nn_forcing_names = list(model_config['nn']['forcings'])
        self._nn_forcing_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._nn_forcing_names)
        }
        # Special keywords that derive from expert outputs.
        self._EXPERT_FORCINGS = {
            'q_prime',
            'q_routed',
            'q_residual',
            'q_abs_residual',
            'q_cv',
            'q_spread',
        }

        # Q normalization for gate inputs.
        self.q_norm = gate_cfg.get('q_norm', 'none')
        self._q_norm_eps = 1e-5
        # For basin_std / nse oracle modes -- set externally from loss_func.
        self._basin_obs_std: Optional[torch.Tensor] = None

        # Per-expert normalization info -- populated by _load_expert.
        self._expert_norm_stats: dict[str, dict] = {}
        self._expert_log_norm_vars: dict[str, list[str]] = {}

        # NN expert output denormalization info -- populated by _load_expert.
        # Maps expert name -> dict with keys: target_mean, target_std,
        # needs_prcp_denorm (bool), prcp_attr_name (str).
        self._nn_expert_denorm: dict[str, dict] = {}

        # MoE's own log_norm setting (from model config).
        self._moe_log_norm_vars: list[str] = (
            model_config.get(
                'use_log_norm',
                [],
            )
            or []
        )

        # Variable names that form xc_nn_norm columns.
        self._xc_nn_var_names: list[str] = list(model_config['nn']['forcings']) + list(
            model_config['nn']['attributes']
        )

        # Load experts
        self.expert_names: list[str] = []
        self.experts = nn.ModuleDict()

        for expert_spec in moe_config['experts']:
            self._load_expert(expert_spec)

        # Build per-expert renormalization transforms.
        self._expert_needs_renorm: dict[str, bool] = {}
        self._build_renorm_transforms()

        # Expert learning rate for fine-tuning. When set, physics experts
        # are unfrozen and trained with this (typically much smaller) LR.
        # NN experts (e.g., LSTM) stay frozen to preserve their
        # denormalization alignment.
        self.expert_lr = moe_config.get('expert_lr', None)
        self._unfrozen_expert_names: list[str] = []

        # Freeze expert parameters
        if self.freeze_experts:
            for expert in self.experts.values():
                for param in expert.parameters():
                    param.requires_grad = False
                expert.eval()
        elif self.expert_lr is not None:
            # Selective unfreezing: physics experts are unfrozen,
            # NN experts (in _nn_expert_denorm) stay frozen.
            for name, expert in self.experts.items():
                if name in self._nn_expert_denorm:
                    # Keep NN experts frozen.
                    for param in expert.parameters():
                        param.requires_grad = False
                    expert.eval()
                    log.info(f"Expert '{name}' kept frozen (NN expert)")
                else:
                    # Unfreeze physics expert for fine-tuning.
                    for param in expert.parameters():
                        param.requires_grad = True
                    self._unfrozen_expert_names.append(name)
                    log.info(f"Expert '{name}' unfrozen (expert_lr={self.expert_lr})")
        else:
            # All experts trainable at main LR (from-scratch training).
            for name, expert in self.experts.items():
                for param in expert.parameters():
                    param.requires_grad = True
                self._unfrozen_expert_names.append(name)
                log.info(f"Expert '{name}' trainable (main LR, from scratch)")

        # Spatial prior flag (read early so gate init can reference it).
        self._use_spatial_prior = moe_config.get('spatial_prior', False)

        # Initialize gate
        n_experts = len(self.expert_names)

        if self.gate_type == 'tcn':
            self._init_tcn_gate(gate_cfg, n_experts)
        elif self.gate_type == 'lstm':
            self._init_lstm_gate(gate_cfg, n_experts)
        elif self.gate_type == 'cnn':
            self._init_cnn_gate(gate_cfg, n_experts)
        else:
            self._init_mlp_gate(gate_cfg, n_experts)

        self.gate.to(device)

        # Spatial prior MLP: attributes -> base logits [N, K].
        if self._use_spatial_prior and self._uses_temporal_forcings:
            n_attr_dim = (
                self.n_gate_attrs if self.n_gate_attrs > 0 else self.n_attributes
            )
            sp_hidden = gate_cfg.get('spatial_prior_hidden', 32)
            sp_dropout = gate_cfg.get('spatial_prior_dropout', 0.3)
            self.spatial_prior_mlp = nn.Sequential(
                nn.Linear(n_attr_dim, sp_hidden),
                nn.ReLU(),
                nn.Dropout(sp_dropout),
                nn.Linear(sp_hidden, n_experts),
            )
            self.spatial_prior_mlp.to(device)
            log.info(
                f"Spatial prior MLP: n_attrs={n_attr_dim}, "
                f"hidden={sp_hidden}, dropout={sp_dropout}"
            )
        else:
            self.spatial_prior_mlp = None

        # Auxiliary losses.
        self.oracle_loss_weight = moe_config.get('oracle_loss_weight', 0.0)
        self.oracle_metric = moe_config.get('oracle_metric', 'mse')
        self.uniform_reg_weight = moe_config.get('uniform_reg_weight', 0.0)

        # Entropy minimization (replaces uniform reg when > 0).
        self.entropy_reg_weight = moe_config.get('entropy_reg_weight', 0.0)

        # Margin-based oracle filtering.
        self.oracle_margin = moe_config.get('oracle_margin', 0.0)

        # Error-prediction mode: gate outputs predicted errors,
        # weights are formed as softmax(-errors/tau).
        self.error_pred_weight = moe_config.get('error_pred_weight', 0.0)
        self._error_pred_mode = self.error_pred_weight > 0

        # Expert specialization: jointly train experts to become
        # complementary specialists via gated gradient routing.
        spec_cfg = moe_config.get('specialization', {})
        self._use_specialization = bool(spec_cfg)
        self.expert_loss_weight = spec_cfg.get('expert_loss_weight', 0.0)
        self.diversity_weight = spec_cfg.get('diversity_weight', 0.0)
        self.load_balance_weight = spec_cfg.get('load_balance_weight', 0.0)
        self._diversity_metric = spec_cfg.get('diversity_metric', 'correlation')
        # WTA sharpening: raise gate weights to this power before
        # weighting per-expert losses (higher -> harder routing).
        self._wta_sharpness = spec_cfg.get('wta_sharpness', 2.0)
        # Quality floor: unweighted per-expert loss to prevent
        # individual experts from degrading below pretrained quality.
        self.quality_floor_weight = spec_cfg.get('quality_floor_weight', 0.0)

        # Phased training: oracle-first curriculum.
        phase_cfg = moe_config.get('phase', {})
        self._use_phased_training = bool(phase_cfg)
        self._phase1_epochs = phase_cfg.get('phase1_epochs', 0)
        self._phase2_epochs = phase_cfg.get('phase2_epochs', 0)
        self._phase2_oracle_end = phase_cfg.get('phase2_oracle_end', 0.2)
        # Current effective weights (updated per-epoch by update_phase).
        self._main_loss_weight = 0.0 if self._use_phased_training else 1.0
        self._oracle_loss_weight_eff = self.oracle_loss_weight
        self._entropy_reg_weight_eff = self.entropy_reg_weight

        # Automatic Weighted Loss (AWL) for multi-task balancing.
        # When enabled, replaces manual loss weights with learned
        # uncertainty-based weighting (Kendall et al. 2018).
        awl_cfg = moe_config.get('awl', {})
        self._use_awl = awl_cfg.get('enabled', False)
        self.awl: Optional[nn.Module] = None
        if self._use_awl:
            from dmg.models.multimodels.auto_weighted_loss import AutoWeightedLoss

            # Count active loss terms: main, oracle, expert_spec, diversity,
            # load_balance, entropy, error_pred.
            self._awl_loss_names: list[str] = ['main']
            if self.oracle_loss_weight > 0:
                self._awl_loss_names.append('oracle')
            if self._use_specialization and self.expert_loss_weight > 0:
                self._awl_loss_names.append('expert_spec')
            if self._use_specialization and self.diversity_weight > 0:
                self._awl_loss_names.append('diversity')
            if self._use_specialization and self.load_balance_weight > 0:
                self._awl_loss_names.append('load_balance')
            if self._use_specialization and self.quality_floor_weight > 0:
                self._awl_loss_names.append('quality_floor')
            if self.entropy_reg_weight > 0:
                self._awl_loss_names.append('entropy')
            if self.error_pred_weight > 0:
                self._awl_loss_names.append('error_pred')

            init_sigmas = awl_cfg.get('init_sigmas', None)
            clamp_range = tuple(awl_cfg.get('clamp_range', [0.1, 10.0]))
            self.awl = AutoWeightedLoss(
                num_losses=len(self._awl_loss_names),
                init_sigmas=init_sigmas,
                clamp_range=clamp_range,
            )
            self.awl.to(device)
            log.info(
                f"AWL enabled: {len(self._awl_loss_names)} losses "
                f"({', '.join(self._awl_loss_names)})"
            )

        # Multiple Choice Learning (MCL) configuration.
        # When enabled, only the best expert per position receives gradient,
        # creating a self-reinforcing specialization loop.
        mcl_cfg = moe_config.get('mcl', {})
        self._use_mcl = mcl_cfg.get('enabled', False)
        self._mcl_granularity = mcl_cfg.get('granularity', 'timestep')
        self._mcl_window_size = mcl_cfg.get('window_size', 30)
        self._mcl_gate_loss_weight = mcl_cfg.get('gate_loss_weight', 1.0)
        self._mcl_expert_loss_weight = mcl_cfg.get('expert_loss_weight', 1.0)
        self._mcl_main_loss_weight = mcl_cfg.get('main_loss_weight', 0.0)
        self._mcl_oracle_metric = mcl_cfg.get('oracle_metric', 'mse')
        # Warmup: number of epochs to train with soft blend only before
        # switching to hard MCL routing.  Gives the gate time to learn
        # the error landscape so winner assignments are stable.
        self._mcl_warmup_epochs = mcl_cfg.get('warmup_epochs', 0)
        # Quality floor: unweighted per-expert loss scaled by this weight.
        # Prevents non-winning experts from degrading on positions they
        # lost, avoiding the oscillation where experts degrade -> win back
        # positions -> degrade again.
        self._mcl_quality_floor_weight = mcl_cfg.get('quality_floor_weight', 0.0)
        # Current epoch (updated by update_phase).
        self._mcl_current_epoch = 0

        # Per-expert loss functions -- set externally by trainer via
        # set_expert_loss_fns(). Maps expert name -> loss function instance.
        self._expert_loss_fns: dict[str, torch.nn.Module] = {}
        # Per-expert loss config (for deferred initialization).
        self._expert_loss_config: dict[str, str] = {}
        if self._use_mcl:
            expert_losses_cfg = mcl_cfg.get('expert_losses', {})
            for ename, loss_name in expert_losses_cfg.items():
                self._expert_loss_config[ename] = loss_name
            log.info(
                f"MCL enabled: granularity={self._mcl_granularity}, "
                f"warmup={self._mcl_warmup_epochs} epochs, "
                f"quality_floor={self._mcl_quality_floor_weight}, "
                f"expert_losses={self._expert_loss_config or 'default (shared)'}"
            )

        # Stored for analysis and oracle loss computation.
        self.gate_weights: Optional[torch.Tensor] = None
        self.gate_logits: Optional[torch.Tensor] = None
        self._expert_streamflow: Optional[torch.Tensor] = None
        self.ensemble_predictions: dict[str, torch.Tensor] = {}

        log.info(
            f"MoE initialized: {n_experts} experts "
            f"({', '.join(self.expert_names)}), "
            f"gate type={self.gate_type}, "
            f"gate mode={self.gate_mode}, "
            f"freeze={self.freeze_experts}"
        )

    def _init_mlp_gate(self, gate_cfg: dict, n_experts: int) -> None:
        """Initialize the MLP gate."""
        n_attr_dim = self.n_gate_attrs if self.n_gate_attrs > 0 else self.n_attributes

        if self._uses_temporal_forcings:
            # New forcings-based temporal mode: expert outputs + standard
            # forcings, with attributes concatenated.
            gate_input_dim = self._compute_temporal_input_dim(n_experts) + n_attr_dim
        else:
            # Legacy mode: static or temporal from xc_nn_norm.
            gate_input_dim = n_attr_dim
            if self.gate_mode == 'temporal':
                gate_input_dim += self.n_forcings

        self.gate = GateMLP(
            n_input=gate_input_dim,
            n_experts=n_experts,
            hidden_size=gate_cfg.get('hidden_size', 128),
            dropout=gate_cfg.get('dropout', 0.1),
        )

        # Break symmetry on final linear layer.
        with torch.no_grad():
            bias = torch.linspace(0.5, -0.5, n_experts)
            self.gate.net[-1].bias.copy_(bias)

        if self._uses_temporal_forcings:
            log.info(
                f"MLP gate (temporal forcings): forcings={self.gate_forcings}, "
                f"n_input={gate_input_dim}, n_attrs={n_attr_dim}"
            )

    def _init_tcn_gate(self, gate_cfg: dict, n_experts: int) -> None:
        """Initialize the TCN gate."""
        from dmg.models.multimodels.gate_tcn import GateTCN

        tcn_input_dim = self._compute_temporal_input_dim(n_experts)

        # Number of attributes for FiLM conditioning.
        if self.n_gate_attrs > 0:
            n_attr_dim = self.n_gate_attrs
        else:
            n_attr_dim = self.n_attributes

        self.gate = GateTCN(
            n_input=tcn_input_dim,
            n_experts=n_experts,
            n_attributes=n_attr_dim,
            width=gate_cfg.get('width', 32),
            depth=gate_cfg.get('depth', 4),
            kernel_size=gate_cfg.get('kernel_size', 3),
            dropout=gate_cfg.get('dropout', 0.1),
            stochastic_depth=gate_cfg.get('stochastic_depth', 0.1),
            use_se=gate_cfg.get('use_se', True),
            multi_scale=gate_cfg.get('multi_scale', True),
        )

        # Break symmetry on the output head's final Conv1d.
        with torch.no_grad():
            bias = torch.linspace(0.5, -0.5, n_experts)
            self.gate.head[-1].bias.copy_(bias)

        log.info(
            f"TCN gate: forcings={self.gate_forcings}, "
            f"n_input={tcn_input_dim}, n_attrs={n_attr_dim}, "
            f"width={gate_cfg.get('width', 32)}, "
            f"depth={gate_cfg.get('depth', 4)}"
        )

    def _init_lstm_gate(self, gate_cfg: dict, n_experts: int) -> None:
        """Initialize the LSTM gate."""
        from dmg.models.multimodels.gate_lstm import GateLSTM

        input_dim = self._compute_temporal_input_dim(n_experts)
        n_attr_dim = self.n_gate_attrs if self.n_gate_attrs > 0 else self.n_attributes

        self.gate = GateLSTM(
            n_input=input_dim,
            n_experts=n_experts,
            n_attributes=n_attr_dim,
            hidden_size=gate_cfg.get('hidden_size', 64),
            num_layers=gate_cfg.get('num_layers', 2),
            dropout=gate_cfg.get('dropout', 0.1),
        )

        # Break symmetry on the output head's final Linear.
        with torch.no_grad():
            bias = torch.linspace(0.5, -0.5, n_experts)
            self.gate.head[-1].bias.copy_(bias)

        log.info(
            f"LSTM gate: forcings={self.gate_forcings}, "
            f"n_input={input_dim}, n_attrs={n_attr_dim}, "
            f"hidden_size={gate_cfg.get('hidden_size', 64)}, "
            f"num_layers={gate_cfg.get('num_layers', 2)}"
        )

    def _init_cnn_gate(self, gate_cfg: dict, n_experts: int) -> None:
        """Initialize the CNN gate."""
        from dmg.models.multimodels.gate_cnn import GateCNN

        input_dim = self._compute_temporal_input_dim(n_experts)
        n_attr_dim = self.n_gate_attrs if self.n_gate_attrs > 0 else self.n_attributes

        self.gate = GateCNN(
            n_input=input_dim,
            n_experts=n_experts,
            n_attributes=n_attr_dim,
            width=gate_cfg.get('width', 32),
            depth=gate_cfg.get('depth', 4),
            kernel_size=gate_cfg.get('kernel_size', 3),
            dropout=gate_cfg.get('dropout', 0.1),
            use_dilation=gate_cfg.get('use_dilation', False),
        )

        # Break symmetry on the output head's final Conv1d.
        with torch.no_grad():
            bias = torch.linspace(0.5, -0.5, n_experts)
            self.gate.head[-1].bias.copy_(bias)

        log.info(
            f"CNN gate: forcings={self.gate_forcings}, "
            f"n_input={input_dim}, n_attrs={n_attr_dim}, "
            f"width={gate_cfg.get('width', 32)}, "
            f"depth={gate_cfg.get('depth', 4)}"
        )

    def _compute_temporal_input_dim(self, n_experts: int) -> int:
        """Compute the temporal input dimension for temporal gates.

        Each entry in ``self.gate_forcings`` contributes a number of
        features depending on its type:

        - ``q_prime``: *K* features (unrouted, one per expert).
        - ``q_routed``: *K* features (routed, one per expert).
        - ``q_residual``: *K*(K-1)/2* features (pairwise differences).
        - ``q_abs_residual``: *K*(K-1)/2* features (absolute differences).
        - ``q_cv``: 1 feature (coefficient of variation across experts).
        - ``q_spread``: 1 feature (std across experts).
        - Any standard forcing name: 1 feature.
        """
        dim = 0
        for name in self.gate_forcings:
            if name in ('q_prime', 'q_routed'):
                dim += n_experts
            elif name in ('q_residual', 'q_abs_residual'):
                dim += n_experts * (n_experts - 1) // 2
            elif name in ('q_cv', 'q_spread'):
                dim += 1
            elif name in self._nn_forcing_idx:
                dim += 1
            else:
                raise ValueError(
                    f"Unknown gate forcing '{name}'. Must be one of "
                    f"{sorted(self._EXPERT_FORCINGS)} or a standard "
                    f"forcing in {self._nn_forcing_names}."
                )
        return dim

    def _normalize_q(
        self,
        q: torch.Tensor,
        dataset_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Normalize expert Q predictions before feeding to gate.

        Parameters
        ----------
        q
            Raw Q tensor ``[T, N, C]``.
        dataset_dict
            Dataset dictionary (for ``basin_std`` mode).

        Returns
        -------
        torch.Tensor
            Normalized Q, same shape.
        """
        if self.q_norm == 'none':
            return q
        elif self.q_norm == 'log':
            # Signed log: preserves sign for residuals (Q_i - Q_j < 0)
            # while compressing magnitude for raw Q.
            return torch.sign(q) * torch.log(q.abs() + self._q_norm_eps)
        elif self.q_norm == 'zscore':
            mean = q.mean(dim=(0, 1), keepdim=True)
            std = q.std(dim=(0, 1), keepdim=True).clamp(min=self._q_norm_eps)
            return (q - mean) / std
        elif self.q_norm == 'basin_std':
            if self._basin_obs_std is None:
                return torch.sign(q) * torch.log(q.abs() + self._q_norm_eps)
            sample_ids = dataset_dict.get('batch_sample')
            if sample_ids is not None:
                std = self._basin_obs_std[sample_ids]  # [N_batch]
                std = std.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            else:
                std = self._basin_obs_std.mean()
            return q / (std.clamp(min=self._q_norm_eps) + 0.1)
        else:
            raise ValueError(f"Unknown q_norm mode: '{self.q_norm}'")

    def anneal_temperature(self, epoch: int) -> None:
        """Update temperature based on annealing schedule.

        Parameters
        ----------
        epoch
            Current training epoch (1-indexed).
        """
        if self._temp_start is None or self._temp_end is None:
            return

        total = self._temp_anneal_epochs or 1
        progress = min(epoch / total, 1.0)

        if self._temp_anneal_schedule == 'cosine':
            import math

            progress = 0.5 * (1 - math.cos(math.pi * progress))

        self.temperature = (
            self._temp_start + (self._temp_end - self._temp_start) * progress
        )

    def update_phase(self, epoch: int) -> None:
        """Update phased training loss weights based on current epoch.

        Phase 1 (epochs 1..phase1_epochs): oracle-only training.
            main_loss_weight = 0, oracle = full weight.
        Phase 2 (epochs phase1_epochs+1..phase1_epochs+phase2_epochs):
            linearly ramp main loss 0->1, decay oracle weight->phase2_oracle_end.
        After both phases: main=1, oracle=phase2_oracle_end.

        Also tracks current epoch for MCL warmup.

        Parameters
        ----------
        epoch
            Current training epoch (1-indexed).
        """
        # Track epoch for MCL warmup.
        self._mcl_current_epoch = epoch

        if not self._use_phased_training:
            self._main_loss_weight = 1.0
            self._oracle_loss_weight_eff = self.oracle_loss_weight
            self._entropy_reg_weight_eff = self.entropy_reg_weight
            return

        p1 = self._phase1_epochs
        p2 = self._phase2_epochs
        oracle_w = self.oracle_loss_weight
        oracle_end = self._phase2_oracle_end
        entropy_w = self.entropy_reg_weight

        if epoch <= p1:
            # Phase 1: oracle-only, no main loss, no entropy reg.
            self._main_loss_weight = 0.0
            self._oracle_loss_weight_eff = oracle_w
            self._entropy_reg_weight_eff = 0.0
        elif epoch <= p1 + p2:
            # Phase 2: ramp main loss, decay oracle, ramp entropy reg.
            progress = (epoch - p1) / max(p2, 1)
            self._main_loss_weight = progress
            self._oracle_loss_weight_eff = oracle_w + (oracle_end - oracle_w) * progress
            self._entropy_reg_weight_eff = entropy_w * progress
        else:
            # Post-phase: full main loss, residual oracle.
            self._main_loss_weight = 1.0
            self._oracle_loss_weight_eff = oracle_end
            self._entropy_reg_weight_eff = entropy_w

        log.debug(
            f"Phase update epoch {epoch}: "
            f"main_w={self._main_loss_weight:.3f}, "
            f"oracle_w={self._oracle_loss_weight_eff:.3f}, "
            f"entropy_w={self._entropy_reg_weight_eff:.3f}"
        )

    def _apply_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply scaling function to gate logits to produce weights.

        Parameters
        ----------
        logits
            Raw gate logits ``[..., K]``.

        Returns
        -------
        torch.Tensor
            Gate weights ``[..., K]``.
        """
        if self.scaling_function == 'softmax':
            return torch.softmax(logits / self.temperature, dim=-1)
        elif self.scaling_function == 'gumbel_softmax':
            if self.training:
                return F.gumbel_softmax(
                    logits,
                    tau=self.temperature,
                    hard=self._gumbel_hard,
                    dim=-1,
                )
            else:
                if self._gumbel_hard:
                    idx = logits.argmax(dim=-1)
                    return F.one_hot(idx, logits.shape[-1]).float()
                return torch.softmax(logits / self.temperature, dim=-1)
        elif self.scaling_function == 'sigmoid':
            return torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown scaling function: {self.scaling_function}")

    def _load_expert(self, expert_spec: dict) -> None:
        """Load a single expert from its config and (optionally) checkpoint.

        Supports both physics-based experts (DplModel) and pure NN
        experts (NnModel).  The expert type is inferred from the config:
        if a ``model.phy`` section exists, a DplModel is created;
        otherwise an NnModel is created.

        When ``checkpoint_path`` is omitted or ``null``, the expert is
        initialized with random weights (for training from scratch).

        Parameters
        ----------
        expert_spec
            Dict with keys: ``name``, ``config_path``.
            Optional: ``checkpoint_path``, ``checkpoint_epoch``,
            ``target_key`` (default ``'streamflow'``).
        """
        name = expert_spec['name']
        config_path = expert_spec['config_path']
        checkpoint_path = expert_spec.get('checkpoint_path', None)
        epoch = expert_spec.get('checkpoint_epoch', None)
        target_key = expert_spec.get('target_key', 'streamflow')

        # Load the expert's YAML config to get its model section.
        # Use OmegaConf to properly coerce types (e.g., 1e-5 -> float).
        expert_raw = OmegaConf.to_container(
            OmegaConf.load(config_path),
            resolve=True,
        )
        expert_model_config = expert_raw['model']

        # Inject top-level settings that Config normally propagates into
        # sub-model dicts (cache_states, warmup).
        cache_states = expert_raw.get('cache_states', False)
        if 'nn' in expert_model_config and expert_model_config['nn']:
            expert_model_config['nn']['cache_states'] = cache_states

        # Handle warmup variants (warmup vs warm_up).
        warmup = expert_model_config.get(
            'warmup',
            expert_model_config.get('warm_up', 0),
        )
        expert_model_config['warmup'] = warmup

        has_physics = (
            'phy' in expert_model_config
            and expert_model_config['phy']
            and 'name' in expert_model_config.get('phy', {})
        )

        if has_physics:
            # Physics-based expert (DplModel).
            if expert_model_config['phy']:
                expert_model_config['phy']['cache_states'] = cache_states
                expert_model_config['phy'].setdefault('warmup', warmup)

            phy_model_name = expert_model_config['phy']['name'][0]
            expert_model = DplModel(
                phy_model_name=phy_model_name,
                config=expert_model_config,
                device=self.device,
            )

            # Checkpoint file path (may be None for from-scratch).
            ckpt_file = (
                os.path.join(checkpoint_path, f"{phy_model_name.lower()}_ep{epoch}.pt")
                if checkpoint_path and epoch
                else None
            )
            model_label = phy_model_name
        else:
            # Pure NN expert (NnModel).
            from dmg.models.wrappers.nn_model import NnModel

            # Determine target names from the expert's train config or
            # fall back to the MoE's target names.
            train_targets = expert_raw.get('train', {}).get('target', [target_key])
            if isinstance(train_targets, str):
                train_targets = [train_targets]

            expert_model = NnModel(
                target_names=train_targets,
                config=expert_model_config,
                device=self.device,
            )

            # Store target key mapping for this NN expert so we can
            # remap its output to 'streamflow' for the gate.
            if not hasattr(self, '_nn_expert_target_map'):
                self._nn_expert_target_map = {}
            self._nn_expert_target_map[name] = dict.fromkeys(train_targets, target_key)

            # Checkpoint file path (may be None for from-scratch).
            nn_name = expert_model_config['nn']['name'].lower()
            ckpt_file = (
                os.path.join(checkpoint_path, f"{nn_name}_ep{epoch}.pt")
                if checkpoint_path and epoch
                else None
            )
            model_label = expert_model_config['nn']['name']

        # Load checkpoint weights (skip if training from scratch).
        if ckpt_file is not None:
            if not os.path.exists(ckpt_file):
                raise FileNotFoundError(f"Expert checkpoint not found: {ckpt_file}")

            state_dict = torch.load(
                ckpt_file,
                weights_only=True,
                map_location=self.device,
            )
            expert_model.load_state_dict(state_dict, strict=False)
        else:
            log.info(f"Expert '{name}' initialized from scratch (no checkpoint)")

        expert_model.to(self.device)

        self.expert_names.append(name)
        self.experts[name] = expert_model

        # Load expert's normalization statistics for per-expert renorm.
        if checkpoint_path is not None:
            norm_stats_path = os.path.join(
                checkpoint_path,
                'normalization_statistics.json',
            )
            if os.path.isfile(norm_stats_path):
                with open(norm_stats_path) as f:
                    self._expert_norm_stats[name] = json.load(f)
            else:
                self._expert_norm_stats[name] = {}
                log.warning(
                    f"Expert '{name}' has no normalization_statistics.json "
                    f"at {norm_stats_path}",
                )
        else:
            # From-scratch: expert uses MoE's normalization directly.
            self._expert_norm_stats[name] = {}

        # Expert's use_log_norm setting.
        self._expert_log_norm_vars[name] = (
            expert_model_config.get('use_log_norm', []) or []
        )

        # NN expert output denormalization: pure NN experts output z-scores
        # of potentially transformed targets. Physics experts output mm/day
        # directly, so only NN experts need this correction.
        if not has_physics and self._expert_norm_stats.get(name):
            # Find the flow target key in the expert's norm stats.
            # The config target name may differ from what was used during
            # training (e.g., config says 'streamflow' but trained on
            # 'runoff'), so search the norm stats directly.
            flow_keys = ['runoff', 'streamflow', 'flow_sim']
            expert_flow_key = None
            for fk in flow_keys:
                if fk in self._expert_norm_stats[name]:
                    expert_flow_key = fk
                    break

            if expert_flow_key is not None:
                # Check if the expert was trained without physics (phy=None).
                # In that case, flow_conversion divides by prcp_mean, making
                # the target dimensionless before z-scoring.
                expert_has_phy = (
                    'phy' in expert_model_config
                    and expert_model_config['phy']
                    and 'name' in expert_model_config.get('phy', {})
                )

                stats = self._expert_norm_stats[name][expert_flow_key]
                # stats = [p10, p90, mean, std]
                denorm_info = {
                    'target_mean': stats[2],
                    'target_std': stats[3],
                    'needs_prcp_denorm': not expert_has_phy,
                    'prcp_attr_name': expert_raw.get(
                        'observations',
                        {},
                    ).get('prcp_mean_name', 'p_mean'),
                }
                self._nn_expert_denorm[name] = denorm_info
                log.info(
                    f"Expert '{name}' output denorm: "
                    f"key='{expert_flow_key}', "
                    f"mean={stats[2]:.4f}, std={stats[3]:.4f}, "
                    f"prcp_denorm={denorm_info['needs_prcp_denorm']}"
                )

        # Column remapping for experts with different forcings/attributes.
        # Build a mapping from expert columns -> MoE columns.
        expert_vars = list(expert_model_config.get('nn', {}).get('forcings', []))
        expert_vars += list(expert_model_config.get('nn', {}).get('attributes', []))
        if expert_vars != self._xc_nn_var_names:
            col_indices = []
            for var in expert_vars:
                if var in self._xc_nn_var_names:
                    col_indices.append(self._xc_nn_var_names.index(var))
                else:
                    log.warning(
                        f"Expert '{name}' requires variable '{var}' which "
                        f"is not in the MoE's input columns."
                    )
                    col_indices.append(-1)  # Sentinel for missing

            if not hasattr(self, '_expert_col_remap'):
                self._expert_col_remap = {}
            self._expert_col_remap[name] = col_indices
            log.info(
                f"Expert '{name}' needs column remapping: "
                f"{len(expert_vars)} expert cols -> "
                f"{len(self._xc_nn_var_names)} MoE cols"
            )

        if ckpt_file is not None:
            log.info(f"Loaded expert '{name}' ({model_label}) from ep{epoch}")
        else:
            log.info(f"Created expert '{name}' ({model_label}) from scratch")

    def _build_renorm_transforms(self) -> None:
        """Pre-compute per-expert renormalization transforms.

        Compares the MoE's normalization statistics with each expert's
        statistics to determine which experts need input re-normalization.
        For columns where both use the same transform type (both linear
        or both log), a simple affine rescaling is pre-computed.
        Columns that differ in log-norm treatment require a nonlinear
        transform at runtime.
        """
        # Load MoE's own normalization statistics.
        moe_norm_stats: dict[str, list[float]] = {}
        if self._model_dir:
            moe_stats_path = os.path.join(
                self._model_dir,
                'normalization_statistics.json',
            )
            if os.path.isfile(moe_stats_path):
                with open(moe_stats_path) as f:
                    moe_norm_stats = json.load(f)

        if not moe_norm_stats:
            # No MoE stats available -- can't do renormalization.
            for name in self.expert_names:
                self._expert_needs_renorm[name] = False
            return

        n_cols = len(self._xc_nn_var_names)

        for name in self.expert_names:
            expert_stats = self._expert_norm_stats.get(name, {})
            expert_log_vars = set(self._expert_log_norm_vars.get(name, []))
            moe_log_vars = set(self._moe_log_norm_vars)

            # Check if any column differs in stats or log-norm treatment.
            needs_renorm = False
            # Per-column: affine scale/offset for linear renorm.
            scale = np.ones(n_cols, dtype=np.float32)
            offset = np.zeros(n_cols, dtype=np.float32)
            # Columns needing nonlinear (log mismatch) transforms.
            nonlinear_cols: list[int] = []
            # For nonlinear cols: store the transform direction.
            # 'to_log': MoE raw -> expert log
            # 'from_log': MoE log -> expert raw
            nonlinear_dir: dict[int, str] = {}
            # Expert stats for nonlinear columns.
            nonlinear_expert_stats: dict[int, tuple[float, float]] = {}
            # MoE stats for nonlinear columns.
            nonlinear_moe_stats: dict[int, tuple[float, float]] = {}

            for col_idx, var in enumerate(self._xc_nn_var_names):
                moe_stat = moe_norm_stats.get(var)
                exp_stat = expert_stats.get(var)

                if moe_stat is None or exp_stat is None:
                    # Missing stats -- assume they match.
                    continue

                moe_mean, moe_std = moe_stat[2], moe_stat[3]
                exp_mean, exp_std = exp_stat[2], exp_stat[3]

                var_in_moe_log = var in moe_log_vars
                var_in_exp_log = var in expert_log_vars

                if var_in_moe_log == var_in_exp_log:
                    # Same transform type -- check if stats differ.
                    if abs(moe_mean - exp_mean) > 1e-6 or abs(moe_std - exp_std) > 1e-6:
                        # Linear rescaling: z_exp = z_moe * (s_m/s_e) + (m_m - m_e)/s_e
                        needs_renorm = True
                        if abs(exp_std) > 1e-10:
                            scale[col_idx] = moe_std / exp_std
                            offset[col_idx] = (moe_mean - exp_mean) / exp_std
                else:
                    # Different transform types -- nonlinear.
                    needs_renorm = True
                    nonlinear_cols.append(col_idx)
                    if var_in_exp_log and not var_in_moe_log:
                        nonlinear_dir[col_idx] = 'to_log'
                    else:
                        nonlinear_dir[col_idx] = 'from_log'
                    nonlinear_expert_stats[col_idx] = (exp_mean, exp_std)
                    nonlinear_moe_stats[col_idx] = (moe_mean, moe_std)

            self._expert_needs_renorm[name] = needs_renorm

            if needs_renorm:
                # Register as buffers so they move with the model.
                self.register_buffer(
                    f'_renorm_scale_{name}',
                    torch.tensor(scale, dtype=torch.float32),
                )
                self.register_buffer(
                    f'_renorm_offset_{name}',
                    torch.tensor(offset, dtype=torch.float32),
                )
                # Store nonlinear column info (not tensors, just metadata).
                setattr(
                    self,
                    f'_renorm_nonlinear_{name}',
                    {
                        'cols': nonlinear_cols,
                        'dirs': nonlinear_dir,
                        'expert_stats': nonlinear_expert_stats,
                        'moe_stats': nonlinear_moe_stats,
                    },
                )

                log.info(
                    f"Expert '{name}' requires input renormalization "
                    f"({len(nonlinear_cols)} nonlinear columns)",
                )

    def _renormalize_for_expert(
        self,
        name: str,
        xc_nn_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Re-normalize xc_nn_norm from MoE space to expert space.

        Parameters
        ----------
        name
            Expert name.
        xc_nn_norm
            Normalized input tensor in MoE normalization space.
            Shape ``[T, N, D]``.

        Returns
        -------
        torch.Tensor
            Re-normalized tensor in the expert's normalization space.
        """
        if not self._expert_needs_renorm.get(name, False):
            return xc_nn_norm

        scale = getattr(self, f'_renorm_scale_{name}').to(xc_nn_norm.device)
        offset = getattr(self, f'_renorm_offset_{name}').to(xc_nn_norm.device)
        nl_info = getattr(self, f'_renorm_nonlinear_{name}')

        # Start with affine rescaling for all columns.
        out = xc_nn_norm * scale + offset

        # Handle nonlinear columns (log-norm mismatch).
        for col in nl_info['cols']:
            direction = nl_info['dirs'][col]
            moe_mean, moe_std = nl_info['moe_stats'][col]
            exp_mean, exp_std = nl_info['expert_stats'][col]

            # Step 1: Denormalize from MoE space to raw.
            z_moe = xc_nn_norm[..., col]
            raw_in_moe_space = z_moe * moe_std + moe_mean

            if direction == 'to_log':
                # MoE used raw normalization, expert used log.
                # raw_in_moe_space is the raw physical value.
                # Apply log10(sqrt(x) + 0.1) then z-score with expert stats.
                raw_val = raw_in_moe_space.clamp(min=0.0)
                transformed = torch.log10(torch.sqrt(raw_val) + 0.1)
                out[..., col] = (transformed - exp_mean) / exp_std
            else:
                # MoE used log normalization, expert used raw.
                # raw_in_moe_space is the log-transformed value.
                # Invert: x = (10^v - 0.1)^2 then z-score with expert stats.
                raw_val = (torch.pow(10.0, raw_in_moe_space) - 0.1) ** 2
                out[..., col] = (raw_val - exp_mean) / exp_std

        return out

    def _build_gate_input(
        self,
        dataset_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Construct MLP gate input based on gate mode and available data.

        Returns
        -------
        torch.Tensor
            ``[N, D]`` for static mode or ``[T, N, D]`` for temporal.
        """
        has_gate_data = 'c_gate_norm' in dataset_dict

        if self.gate_mode == 'temporal':
            # Forcings from xc_nn_norm: [T, N, n_forc]
            forcings = dataset_dict['xc_nn_norm'][:, :, : self.n_forcings]
            T, N = forcings.shape[:2]

            if has_gate_data:
                # Tile static gate attrs across time: [N, D] -> [T, N, D]
                c_gate = dataset_dict['c_gate_norm']
                c_gate_tiled = c_gate.unsqueeze(0).expand(T, -1, -1)
            else:
                # Fallback: use nn attributes from xc_nn_norm.
                attrs = dataset_dict['xc_nn_norm'][:, :, self.n_forcings :]
                c_gate_tiled = attrs

            return torch.cat([forcings, c_gate_tiled], dim=-1)  # [T, N, D]

        else:  # static
            if has_gate_data:
                return dataset_dict['c_gate_norm']  # [N, D]
            else:
                # Fallback: attributes from xc_nn_norm at t=0.
                return dataset_dict['xc_nn_norm'][0, :, self.n_forcings :]

    def _build_temporal_gate_input(
        self,
        expert_outputs: dict[str, dict[str, torch.Tensor]],
        dataset_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Construct TCN gate input from ``gate.forcings`` spec.

        Each entry in ``self.gate_forcings`` is either a special keyword
        (``q_prime``, ``q_routed``, ``q_residual``) that derives
        features from expert outputs, or a standard forcing name looked
        up by column index in ``xc_nn_norm``.

        Parameters
        ----------
        expert_outputs
            Dict mapping expert names to their output dicts.
        dataset_dict
            Dataset dictionary from the sampler.

        Returns
        -------
        x
            Temporal input ``[N, T, n_input]`` (batch-first for TCN).
        a
            Static attributes ``[N, D_attr]`` for FiLM, or ``None``.
        """
        parts: list[torch.Tensor] = []

        # Expert outputs may be shorter than xc_nn_norm because physics
        # models trim the warmup period.  Determine the expert output
        # length so we can align standard forcings.
        first_expert = self.expert_names[0]
        T_expert = expert_outputs[first_expert]['streamflow'].shape[0]

        for name in self.gate_forcings:
            if name == 'q_routed':
                # Routed streamflow: each [T_expert, N, 1] -> cat
                q_list = [expert_outputs[n]['streamflow'] for n in self.expert_names]
                q_cat = torch.cat(q_list, dim=-1)
                parts.append(self._normalize_q(q_cat, dataset_dict))

            elif name == 'q_prime':
                # Unrouted streamflow: each [T_expert, N, 1] -> cat.
                # Falls back to routed streamflow for pure NN experts.
                q_list = [
                    expert_outputs[n].get(
                        'streamflow_no_rout',
                        expert_outputs[n]['streamflow'],
                    )
                    for n in self.expert_names
                ]
                q_cat = torch.cat(q_list, dim=-1)
                parts.append(self._normalize_q(q_cat, dataset_dict))

            elif name == 'q_residual':
                # Pairwise differences between expert routed predictions.
                q_list = [
                    expert_outputs[n]['streamflow'].squeeze(-1)
                    for n in self.expert_names
                ]
                diffs = []
                for i in range(len(q_list)):
                    for j in range(i + 1, len(q_list)):
                        diffs.append(q_list[i] - q_list[j])
                q_res = torch.stack(diffs, dim=-1)
                parts.append(self._normalize_q(q_res, dataset_dict))

            elif name == 'q_abs_residual':
                # Absolute pairwise differences (magnitude of disagreement).
                q_list = [
                    expert_outputs[n]['streamflow'].squeeze(-1)
                    for n in self.expert_names
                ]
                diffs = []
                for i in range(len(q_list)):
                    for j in range(i + 1, len(q_list)):
                        diffs.append((q_list[i] - q_list[j]).abs())
                q_abs = torch.stack(diffs, dim=-1)
                parts.append(self._normalize_q(q_abs, dataset_dict))

            elif name == 'q_cv':
                # Coefficient of variation across experts (strongest
                # oracle predictor: Spearman r = 0.26).
                q_list = [
                    expert_outputs[n]['streamflow'].squeeze(-1)
                    for n in self.expert_names
                ]
                q_stack = torch.stack(q_list, dim=0)  # [K, T, N]
                q_std = q_stack.std(dim=0)  # [T, N]
                q_mean = q_stack.mean(dim=0).abs().clamp(min=1e-5)
                cv = (q_std / q_mean).unsqueeze(-1)  # [T, N, 1]
                parts.append(self._normalize_q(cv, dataset_dict))

            elif name == 'q_spread':
                # Standard deviation across expert predictions.
                q_list = [
                    expert_outputs[n]['streamflow'].squeeze(-1)
                    for n in self.expert_names
                ]
                q_stack = torch.stack(q_list, dim=0)  # [K, T, N]
                spread = q_stack.std(dim=0).unsqueeze(-1)  # [T, N, 1]
                parts.append(self._normalize_q(spread, dataset_dict))

            else:
                # Standard forcing -- look up column in xc_nn_norm and
                # trim to match expert output length (warmup removed).
                col = self._nn_forcing_idx[name]
                forcing = dataset_dict['xc_nn_norm'][:, :, col : col + 1]
                parts.append(forcing[-T_expert:])

        x = torch.cat(parts, dim=-1)  # [T_expert, N, D]
        x = x.permute(1, 0, 2)  # [N, T, D]  (batch-first for TCN)

        # Static attributes for FiLM conditioning.
        a = None
        if self.n_gate_attrs > 0 and 'c_gate_norm' in dataset_dict:
            a = dataset_dict['c_gate_norm']  # [N, D_attr]
        elif self.n_attributes > 0:
            a = dataset_dict['xc_nn_norm'][0, :, self.n_forcings :]

        return x, a

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run frozen experts, gate, and weighted combination.

        Parameters
        ----------
        dataset_dict
            Dataset dictionary from the sampler containing
            ``xc_nn_norm``, ``x_phy``, ``target``, and optionally
            ``c_gate_norm``.

        Returns
        -------
        dict[str, torch.Tensor]
            Combined predictions dict (e.g. ``{'streamflow': [T, N, 1]}``).
        """
        expert_outputs: dict[str, dict[str, torch.Tensor]] = {}

        # 1. Run each frozen expert (no gradients if frozen).
        #    Per-expert renormalization handles cases where experts were
        #    trained with different normalization (e.g. use_log_norm).
        for name in self.expert_names:
            expert = self.experts[name]
            col_remap = getattr(self, '_expert_col_remap', {})
            if name in col_remap:
                # Expert has different column layout -- remap.
                expert_data = dict(dataset_dict)
                indices = col_remap[name]
                valid_indices = [i for i in indices if i >= 0]
                expert_data['xc_nn_norm'] = dataset_dict['xc_nn_norm'][
                    ..., valid_indices
                ]
            elif self._expert_needs_renorm.get(name, False):
                expert_data = dict(dataset_dict)
                expert_data['xc_nn_norm'] = self._renormalize_for_expert(
                    name,
                    dataset_dict['xc_nn_norm'],
                )
            else:
                expert_data = dataset_dict
            # Run with gradients only for unfrozen experts.
            is_frozen = self.freeze_experts or (
                self.expert_lr and name not in self._unfrozen_expert_names
            )
            if is_frozen:
                with torch.no_grad():
                    expert_outputs[name] = expert(expert_data)
            else:
                expert_outputs[name] = expert(expert_data)

            # Remap NN expert output keys (e.g., runoff -> streamflow).
            nn_map = getattr(self, '_nn_expert_target_map', {})
            if name in nn_map:
                remapped = {}
                for old_key, new_key in nn_map[name].items():
                    if old_key in expert_outputs[name]:
                        remapped[new_key] = expert_outputs[name][old_key]
                # Keep any keys that weren't remapped.
                for k, v in expert_outputs[name].items():
                    if k not in nn_map[name]:
                        remapped[k] = v
                expert_outputs[name] = remapped

            # Denormalize NN expert output from z-scores -> mm/day.
            # Pure NN experts output z-scores of (possibly dimensionless)
            # target. Physics experts output mm/day directly.
            if name in self._nn_expert_denorm:
                denorm = self._nn_expert_denorm[name]
                for key in list(expert_outputs[name].keys()):
                    pred = expert_outputs[name][key]
                    # z-score -> physical: pred * std + mean
                    pred = pred * denorm['target_std'] + denorm['target_mean']
                    # If expert was trained without physics model, the
                    # target was made dimensionless by dividing by
                    # prcp_mean. Reverse that to get mm/day.
                    if denorm['needs_prcp_denorm']:
                        attr_name = denorm['prcp_attr_name']
                        nn_attrs = list(self.model_config['nn']['attributes'])
                        if attr_name in nn_attrs:
                            attr_idx = nn_attrs.index(attr_name)
                            # c_nn is raw (unnormalized) attributes [N, D].
                            prcp_mean = dataset_dict['c_nn'][:, attr_idx]
                            # Expand to match pred shape [T, N, ...]
                            prcp_mean = prcp_mean.unsqueeze(0)
                            while prcp_mean.ndim < pred.ndim:
                                prcp_mean = prcp_mean.unsqueeze(-1)
                            pred = pred * prcp_mean
                    expert_outputs[name][key] = pred

        # 2. Gate forward (with gradients).
        if self.scaling_function == 'uniform':
            # Simple average -- no gate needed.
            N = dataset_dict['xc_nn_norm'].shape[1]
            K = len(self.expert_names)
            weights = torch.full(
                (N, K),
                1.0 / K,
                device=self.device,
                dtype=torch.float32,
            )
        elif self._uses_temporal_forcings:
            # Temporal gate: TCN, LSTM, CNN, or MLP with forcings.
            x_gate, a_gate = self._build_temporal_gate_input(
                expert_outputs,
                dataset_dict,
            )

            if self.gate_type == 'mlp':
                # MLP: concatenate attrs and transpose to time-first.
                if a_gate is not None:
                    a_tiled = a_gate.unsqueeze(1).expand(-1, x_gate.size(1), -1)
                    x_gate = torch.cat([x_gate, a_tiled], dim=-1)
                gate_logits = self.gate(
                    x_gate.permute(1, 0, 2),  # [T, N, D]
                )  # [T, N, K]
            else:
                # TCN/LSTM/CNN: batch-first, handle attrs internally.
                gate_logits = self.gate(
                    x_gate,
                    a=a_gate,
                ).permute(1, 0, 2)  # [N, T, K] -> [T, N, K]

            # Spatial prior: add per-basin base logits from attributes.
            if self.spatial_prior_mlp is not None and a_gate is not None:
                spatial_logits = self.spatial_prior_mlp(a_gate)  # [N, K]
                gate_logits = gate_logits + spatial_logits.unsqueeze(0)  # [T, N, K]

            # Error-prediction mode: gate outputs = predicted errors.
            # Negate to form preference logits (lower error = higher weight).
            if self._error_pred_mode:
                self._predicted_errors = gate_logits  # store for aux loss
                weights = self._apply_scaling(-gate_logits)
            else:
                weights = self._apply_scaling(gate_logits)
        else:
            # Legacy MLP gate: uses forcings/attributes directly.
            gate_input = self._build_gate_input(dataset_dict)
            gate_logits = self.gate(gate_input)  # [N, K] or [T, N, K]
            if self._error_pred_mode:
                self._predicted_errors = gate_logits
                weights = self._apply_scaling(-gate_logits)
            else:
                weights = self._apply_scaling(gate_logits)

        self.gate_weights = weights
        # Store gate logits for oracle loss (None for uniform).
        # In error-pred mode, negate for oracle CE compatibility.
        if self._error_pred_mode:
            self.gate_logits = (
                -gate_logits if self.scaling_function != 'uniform' else None
            )
        else:
            self.gate_logits = (
                gate_logits if self.scaling_function != 'uniform' else None
            )

        # Store expert streamflow predictions for oracle, error-pred,
        # specialization, and MCL losses.
        if (
            self.oracle_loss_weight > 0
            or self._error_pred_mode
            or self._use_specialization
            or self._use_mcl
        ):
            self._expert_streamflow = torch.stack(
                [
                    expert_outputs[n]['streamflow'].squeeze(-1)
                    for n in self.expert_names
                ],
                dim=0,
            )  # [K, T, N]
        else:
            self._expert_streamflow = None

        # Store batch_sample for NSE oracle metric.
        self._batch_sample = dataset_dict.get('batch_sample')

        # 3. Weighted combination of expert predictions.
        predictions_list = [expert_outputs[n] for n in self.expert_names]
        shared_keys = find_shared_keys(*predictions_list)

        # Reshape weights for broadcasting.
        # Static:   [N, K] -> [K, 1, N] (broadcast over T)
        # Temporal:  [T, N, K] -> [K, T, N]
        if weights.ndim == 2:
            w_base = weights.T.unsqueeze(1)
        else:
            w_base = weights.permute(2, 0, 1)

        combined: dict[str, torch.Tensor] = {}
        for key in shared_keys:
            preds = []
            for n in self.expert_names:
                p = expert_outputs[n].get(key)
                if p is None:
                    continue
                preds.append(p)

            if not preds:
                combined[key] = None
                continue

            # Stack: [K, T, N, ...] and weight by gate.
            pred_stack = torch.stack(preds, dim=0)

            # Expand w_base trailing dims to match pred_stack.
            w = w_base
            while w.ndim < pred_stack.ndim:
                w = w.unsqueeze(-1)

            combined[key] = (pred_stack * w).sum(dim=0)

        self.ensemble_predictions = combined
        return combined

    def compute_oracle_loss(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute oracle-guided cross-entropy auxiliary loss.

        At each valid (timestep, basin), identifies which expert had the
        lowest error relative to the target and produces a cross-entropy
        loss that pushes the gate logits toward selecting that expert.

        The error metric used for oracle label selection is controlled by
        ``self.oracle_metric``:

        - ``'mse'``: Squared error ``(pred - obs)^2``.  Default.  Biased
          toward high-flow accuracy since absolute errors scale with
          magnitude.
        - ``'relative'``: Relative absolute error
          ``|pred - obs| / (|obs| + eps)``.  Treats errors equally across
          flow magnitudes, improving both high- and low-flow selection.
        - ``'log'``: Log-space squared error
          ``(log(pred + eps) - log(obs + eps))^2``.  Strongly emphasises
          low-flow accuracy.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``.

        Returns
        -------
        torch.Tensor
            Scalar oracle cross-entropy loss (zero if not applicable).
        """
        if self.gate_logits is None or self._expert_streamflow is None:
            return torch.tensor(0.0, device=self.device)

        target = target.squeeze(-1)  # [T, N]
        K, T_exp, N = self._expert_streamflow.shape

        # Align target to expert output length (warmup trimmed).
        T_target = target.shape[0]
        if T_target > T_exp:
            target = target[-T_exp:]

        # Gate logits may also need alignment.
        logits = self.gate_logits  # [T, N, K]  (temporal) or [N, K] (static)
        if logits.ndim == 2:
            # Static gate -- broadcast to [T, N, K] for per-timestep oracle.
            logits = logits.unsqueeze(0).expand(T_exp, -1, -1)
        elif logits.shape[0] > T_exp:
            logits = logits[-T_exp:]

        # Compute per-expert error based on oracle_metric.
        preds = self._expert_streamflow  # [K, T, N]
        obs = target.unsqueeze(0)  # [1, T, N]

        if self.oracle_metric == 'relative':
            errors = (preds - obs).abs() / (obs.abs() + 1e-5)
        elif self.oracle_metric == 'log':
            eps = 1e-5
            errors = (
                torch.log(preds.clamp(min=eps)) - torch.log(obs.clamp(min=eps))
            ) ** 2
        elif self.oracle_metric == 'nse':
            # NSE-aligned: squared error normalized by per-basin obs std,
            # matching the NseBatchLoss gradient scaling.
            errors = (preds - obs) ** 2
            if self._basin_obs_std is not None and self._batch_sample is not None:
                std = self._basin_obs_std[self._batch_sample]  # [N]
                std = std.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
                errors = errors / (std + 0.1) ** 2
        else:  # 'mse' (default)
            errors = (preds - obs) ** 2

        # Mask invalid (NaN) targets.
        valid = ~torch.isnan(target)  # [T, N]

        # Best expert at each valid position.
        errors_masked = errors.clone()
        errors_masked[:, ~valid] = float('inf')
        oracle_labels = errors_masked.argmin(dim=0)  # [T, N]

        # Margin-based filtering: only keep positions where the best
        # expert is significantly better than the second-best.
        if self.oracle_margin > 0:
            sorted_errors, _ = errors_masked.sort(dim=0)  # [K, T, N]
            best_err = sorted_errors[0]  # [T, N]
            second_err = sorted_errors[1]  # [T, N]
            # Relative margin: (second - best) / (second + eps)
            margin = (second_err - best_err) / (second_err + 1e-8)
            clear_winner = margin > self.oracle_margin  # [T, N]
            # Combine with valid mask.
            valid = valid & clear_winner

        # Flatten valid positions.
        logits_flat = logits[valid]  # [M, K]
        labels_flat = oracle_labels[valid]  # [M]

        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        return F.cross_entropy(logits_flat, labels_flat)

    def compute_uniform_reg(self) -> torch.Tensor:
        """Compute KL divergence from gate weights to uniform distribution.

        Penalises the gate for deviating from a uniform weighting of
        experts, ensuring it only shifts weight when there is strong
        evidence.  This prevents the gate from doing worse than a simple
        average.

        Returns
        -------
        torch.Tensor
            Scalar KL(gate_weights || uniform).
        """
        if self.gate_weights is None:
            return torch.tensor(0.0, device=self.device)

        weights = self.gate_weights  # [T, N, K] or [N, K]
        K = weights.shape[-1]

        # Uniform target distribution.
        uniform = torch.full_like(weights, 1.0 / K)

        # KL(weights || uniform) = sum(w * log(w / u))
        # Clamp to avoid log(0).
        log_ratio = torch.log(weights.clamp(min=1e-8) / uniform)
        kl = (weights * log_ratio).sum(dim=-1)  # [...] per position

        # Average over all valid positions.
        return kl.mean()

    def compute_error_pred_loss(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute error-prediction auxiliary loss.

        Trains the gate to predict each expert's squared error at each
        timestep (regression).  The predicted errors then naturally form
        good weights via ``softmax(-predicted_errors)``.

        This loss is: MSE(predicted_errors, actual_errors), where
        actual_errors are normalized per-basin squared errors.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``.

        Returns
        -------
        torch.Tensor
            Scalar error-prediction MSE loss.
        """
        if not hasattr(self, '_predicted_errors') or self._predicted_errors is None:
            return torch.tensor(0.0, device=self.device)
        if self._expert_streamflow is None:
            return torch.tensor(0.0, device=self.device)

        target = target.squeeze(-1)  # [T, N]
        K, T_exp, N = self._expert_streamflow.shape

        # Align target
        if target.shape[0] > T_exp:
            target = target[-T_exp:]

        pred_errors = self._predicted_errors  # [T, N, K]
        if pred_errors.shape[0] > T_exp:
            pred_errors = pred_errors[-T_exp:]

        # Actual per-expert errors (normalized by basin std).
        preds = self._expert_streamflow  # [K, T, N]
        obs = target.unsqueeze(0)  # [1, T, N]
        actual_errors = (preds - obs) ** 2  # [K, T, N]

        # Normalize by basin obs std if available.
        if self._basin_obs_std is not None and self._batch_sample is not None:
            std = self._basin_obs_std[self._batch_sample]  # [N]
            std = std.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            actual_errors = actual_errors / (std + 0.1) ** 2

        # Rearrange to [T, N, K] to match pred_errors.
        actual_errors = actual_errors.permute(1, 2, 0)  # [T, N, K]

        # Log-transform to compress range (errors can span orders of magnitude).
        actual_log = torch.log(actual_errors.clamp(min=1e-8))
        pred_log = pred_errors  # gate outputs are unbounded, interpret as log-errors

        # Mask invalid targets.
        valid = ~torch.isnan(target)  # [T, N]
        valid_3d = valid.unsqueeze(-1).expand_as(actual_log)

        if valid_3d.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        return F.mse_loss(pred_log[valid_3d], actual_log[valid_3d].detach())

    def compute_entropy_reg(self) -> torch.Tensor:
        """Compute entropy of gate weights (to be *minimized*).

        Low entropy -> peaky/sharp weight distributions -> decisive selection.
        High entropy -> uniform weights -> averaging.

        Minimizing this encourages the gate to commit to one expert rather
        than hedging across all of them.

        Returns
        -------
        torch.Tensor
            Scalar mean entropy of gate weights.
        """
        if self.gate_weights is None:
            return torch.tensor(0.0, device=self.device)

        weights = self.gate_weights  # [T, N, K] or [N, K]
        # H = -sum(w * log(w))
        entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=-1)
        return entropy.mean()

    # ------------------------------------------------------------------
    # Expert specialization losses
    # ------------------------------------------------------------------

    def compute_expert_specialization_loss(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Winner-Take-All expert loss for specialization.

        Each expert's per-sample squared error is weighted by its
        sharpened gate weight.  This focuses each expert's gradient on
        the samples it is "assigned" to, creating a positive feedback
        loop toward specialization.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``.

        Returns
        -------
        torch.Tensor
            Scalar WTA expert loss.
        """
        if self._expert_streamflow is None or self.gate_weights is None:
            return torch.tensor(0.0, device=self.device)

        target = target.squeeze(-1)  # [T, N]
        K, T_exp, N = self._expert_streamflow.shape

        # Align target to expert output length.
        if target.shape[0] > T_exp:
            target = target[-T_exp:]

        # Align gate weights.
        weights = self.gate_weights  # [T, N, K] or [N, K]
        if weights.ndim == 2:
            weights = weights.unsqueeze(0).expand(T_exp, -1, -1)
        elif weights.shape[0] > T_exp:
            weights = weights[-T_exp:]

        # Sharpen weights to approximate hard routing while keeping
        # differentiability.  Higher sharpness -> more WTA-like.
        w_sharp = weights**self._wta_sharpness  # [T, N, K]
        w_sharp = w_sharp / (w_sharp.sum(dim=-1, keepdim=True) + 1e-8)

        # Per-expert squared error.
        obs = target.unsqueeze(0)  # [1, T, N]
        errors = (self._expert_streamflow - obs) ** 2  # [K, T, N]
        # Clamp extreme errors to prevent NaN from runaway experts.
        errors = errors.clamp(max=1e6)

        # Normalize by basin obs std if available (matches NSE scaling).
        if self._basin_obs_std is not None and self._batch_sample is not None:
            std = self._basin_obs_std[self._batch_sample]  # [N]
            std = std.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            errors = errors / (std + 0.1) ** 2

        # Mask invalid targets and non-finite errors.
        valid = ~torch.isnan(target) & torch.isfinite(errors).all(dim=0)
        errors[:, ~valid] = 0.0

        # Weight each expert's error by its (sharpened) gate weight.
        # w_sharp: [T, N, K] -> [K, T, N]
        w_perm = w_sharp.permute(2, 0, 1)  # [K, T, N]
        weighted_errors = (w_perm * errors).sum(dim=(1, 2))  # [K]

        n_valid = valid.sum().clamp(min=1).float()
        return weighted_errors.sum() / (K * n_valid)

    def compute_diversity_loss(
        self,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Penalize correlated expert ERROR PATTERNS.

        Instead of decorrelating predictions (which forces experts to
        be wrong in arbitrary ways), decorrelate their *errors*
        relative to the observation. This encourages each expert to
        fail in different situations -- exactly what multimodeling
        exploits -- without degrading overall individual accuracy.

        Falls back to prediction-correlation if no target is available.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``, or ``None``.

        Returns
        -------
        torch.Tensor
            Scalar diversity penalty (higher = more correlated = bad).
        """
        if self._expert_streamflow is None:
            return torch.tensor(0.0, device=self.device)

        K, T, N = self._expert_streamflow.shape
        if K < 2:
            return torch.tensor(0.0, device=self.device)

        if target is not None:
            # Error-pattern diversity (preferred).
            target = target.squeeze(-1)
            if target.shape[0] > T:
                target = target[-T:]
            obs = target.unsqueeze(0)  # [1, T, N]
            # Per-expert signed error: [K, T, N]
            errors = self._expert_streamflow - obs
            signals = errors.reshape(K, -1)
        else:
            # Fallback: prediction correlation.
            signals = self._expert_streamflow.reshape(K, -1)

        # Remove NaN/Inf positions across ALL experts.
        valid_mask = torch.isfinite(signals).all(dim=0)
        signals = signals[:, valid_mask]

        if signals.shape[1] < 10:
            return torch.tensor(0.0, device=self.device)

        # Detach to avoid unstable second-order gradients through
        # the correlation computation.  The WTA expert loss provides
        # the direct gradient signal to push experts apart; this loss
        # serves as a regularizer for the gate.
        signals = signals.detach()

        # Center each expert's signal.
        signals_centered = signals - signals.mean(dim=1, keepdim=True)

        # Pairwise Pearson correlation.
        norms = signals_centered.norm(dim=1, keepdim=True).clamp(min=1e-6)
        signals_normed = signals_centered / norms  # [K, M]
        corr_matrix = signals_normed @ signals_normed.T  # [K, K]

        # Mean of upper-triangle (excluding diagonal).
        mask = torch.triu(torch.ones(K, K, device=self.device), diagonal=1)
        n_pairs = mask.sum()
        mean_corr = (corr_matrix * mask).sum() / n_pairs.clamp(min=1)

        return mean_corr.clamp(min=0.0)

    def compute_expert_quality_loss(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert quality floor: prevent experts from degrading.

        Computes the average per-expert NSE-like loss (unweighted by
        gate) and returns it.  When combined with the main gated loss,
        this ensures each expert maintains reasonable accuracy on the
        full dataset, not just its gate-assigned niche.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``.

        Returns
        -------
        torch.Tensor
            Average per-expert MSE (NSE-normalized).
        """
        if self._expert_streamflow is None:
            return torch.tensor(0.0, device=self.device)

        target = target.squeeze(-1)  # [T, N]
        K, T_exp, N = self._expert_streamflow.shape

        if target.shape[0] > T_exp:
            target = target[-T_exp:]

        obs = target.unsqueeze(0)  # [1, T, N]
        errors = (self._expert_streamflow - obs) ** 2  # [K, T, N]
        errors = errors.clamp(max=1e6)

        # Normalize by basin obs std if available.
        if self._basin_obs_std is not None and self._batch_sample is not None:
            std = self._basin_obs_std[self._batch_sample]
            std = std.unsqueeze(0).unsqueeze(0)
            errors = errors / (std + 0.1) ** 2

        # Mask invalid targets.
        valid = ~torch.isnan(target)
        errors[:, ~valid] = 0.0

        n_valid = valid.sum().clamp(min=1).float()
        return errors.sum() / (K * n_valid)

    def compute_load_balance_loss(self) -> torch.Tensor:
        """Load balance loss (Switch Transformer style).

        Penalizes uneven expert utilization to prevent mode collapse
        (one expert getting all the weight).

        L_balance = K * sum_k(f_k^2)

        where f_k is the mean gate weight for expert k.  Minimized
        when all f_k = 1/K (uniform utilization).

        Returns
        -------
        torch.Tensor
            Scalar load balance penalty.
        """
        if self.gate_weights is None:
            return torch.tensor(0.0, device=self.device)

        weights = self.gate_weights  # [T, N, K] or [N, K]
        K = weights.shape[-1]

        # Mean weight per expert across all positions.
        f = weights.mean(dim=tuple(range(weights.ndim - 1)))  # [K]

        # Switch Transformer load balance: K * sum(f_k^2)
        # Minimum at f_k = 1/K for all k -> loss = 1.0
        return K * (f**2).sum()

    # ------------------------------------------------------------------
    # Multiple Choice Learning (MCL)
    # ------------------------------------------------------------------

    def set_expert_loss_fns(
        self,
        loss_fns: dict[str, torch.nn.Module],
    ) -> None:
        """Set per-expert loss functions for MCL training.

        Called by the trainer after loss function instantiation.

        Parameters
        ----------
        loss_fns
            Maps expert name -> loss function instance.
        """
        self._expert_loss_fns = loss_fns
        log.info(
            f"MCL expert loss functions set: "
            f"{{{', '.join(f'{k}: {v.name}' for k, v in loss_fns.items())}}}"
        )

    def compute_mcl_loss(
        self,
        target: torch.Tensor,
        shared_loss_fn: torch.nn.Module,
        batch_sample: np.ndarray,
    ) -> torch.Tensor:
        """Multiple Choice Learning: only the best expert gets gradient.

        1. Determine the winner at each valid position using a common
           error metric (detached -- no gradient through winner selection).
        2. For each expert, compute its loss ONLY on positions where it
           won.  Per-expert loss functions are used if configured,
           otherwise the shared loss function is used.
        3. Gate is trained with oracle cross-entropy to predict winners.
        4. Quality floor prevents non-winning experts from degrading.

        During the warmup phase (first ``mcl.warmup_epochs``), falls back
        to soft blend training (main loss on gated combination + oracle
        CE for the gate).  This lets the gate learn the error landscape
        before hard MCL routing kicks in, stabilizing winner assignments.

        Parameters
        ----------
        target
            Ground truth ``[T, N]`` or ``[T, N, 1]``.
        shared_loss_fn
            The main loss function (used for experts without a
            per-expert loss and for the gated ensemble output).
        batch_sample
            Basin indices for the current batch.

        Returns
        -------
        torch.Tensor
            Combined MCL loss (expert winner loss + gate oracle loss +
            optional quality floor + optional main blend loss).
        """
        if self._expert_streamflow is None:
            return torch.tensor(0.0, device=self.device)

        target = target.squeeze(-1)  # [T, N]
        K, T_exp, N = self._expert_streamflow.shape

        # Align target to expert output length (warmup trimmed).
        if target.shape[0] > T_exp:
            target = target[-T_exp:]

        valid = ~torch.isnan(target)  # [T, N]
        sample_ids = batch_sample.astype(int) if batch_sample is not None else None

        # ---- Step 1: Determine winners (no gradient) ----
        with torch.no_grad():
            preds = self._expert_streamflow  # [K, T, N]
            obs = target.unsqueeze(0)  # [1, T, N]

            if self._mcl_oracle_metric == 'nse':
                errors = (preds - obs) ** 2
                if self._basin_obs_std is not None and self._batch_sample is not None:
                    std = self._basin_obs_std[self._batch_sample]
                    std = std.unsqueeze(0).unsqueeze(0)
                    errors = errors / (std + 0.1) ** 2
            elif self._mcl_oracle_metric == 'log':
                eps = 1e-5
                errors = (
                    torch.log(preds.clamp(min=eps)) - torch.log(obs.clamp(min=eps))
                ) ** 2
            else:  # 'mse'
                errors = (preds - obs) ** 2

            errors[:, ~valid] = float('inf')

            if self._mcl_granularity == 'timestep':
                winners = errors.argmin(dim=0)  # [T, N]
            elif self._mcl_granularity == 'window':
                W = self._mcl_window_size
                winners = torch.zeros(T_exp, N, dtype=torch.long, device=self.device)
                for t_start in range(0, T_exp, W):
                    t_end = min(t_start + W, T_exp)
                    window_err = errors[:, t_start:t_end, :].sum(dim=1)
                    win = window_err.argmin(dim=0)
                    winners[t_start:t_end] = win.unsqueeze(0)
            else:  # 'basin'
                basin_err = errors.sum(dim=1)
                win = basin_err.argmin(dim=0)
                winners = win.unsqueeze(0).expand(T_exp, -1)

        # ---- Warmup phase: soft blend only ----
        in_warmup = (
            self._mcl_warmup_epochs > 0
            and self._mcl_current_epoch <= self._mcl_warmup_epochs
        )

        if in_warmup:
            # During warmup: train gate with oracle CE + main blend loss.
            # No hard MCL routing -- experts get gradient through soft blend.
            gate_loss = torch.tensor(0.0, device=self.device)
            if self.gate_logits is not None:
                logits = self.gate_logits
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0).expand(T_exp, -1, -1)
                elif logits.shape[0] > T_exp:
                    logits = logits[-T_exp:]
                logits_flat = logits[valid]
                labels_flat = winners[valid]
                if logits_flat.numel() > 0:
                    gate_loss = F.cross_entropy(logits_flat, labels_flat)

            main_loss = torch.tensor(0.0, device=self.device)
            pred_combined = self.ensemble_predictions.get('streamflow')
            if pred_combined is not None:
                main_loss = shared_loss_fn(
                    pred_combined.squeeze(),
                    target,
                    sample_ids=batch_sample,
                )

            return main_loss + self._mcl_gate_loss_weight * gate_loss

        # ---- Step 2: Winner-only expert loss ----
        expert_loss = torch.tensor(0.0, device=self.device)

        for k, name in enumerate(self.expert_names):
            mask_k = (winners == k) & valid  # [T, N]
            if mask_k.sum() == 0:
                continue

            pred_k = self._expert_streamflow[k]  # [T, N]
            loss_fn = self._expert_loss_fns.get(name, shared_loss_fn)

            p_sub = pred_k[mask_k]
            t_sub = target[mask_k]

            if hasattr(loss_fn, 'std') and sample_ids is not None:
                n_timesteps = target.shape[0]
                std_vals = np.tile(loss_fn.std[sample_ids].T, (n_timesteps, 1))
                std_batch = torch.tensor(
                    std_vals,
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                )
                std_sub = std_batch[mask_k]
                eps = getattr(loss_fn, 'eps', 0.1)

                if hasattr(loss_fn, 'eps_log'):
                    p_sub = torch.log(p_sub.clamp(min=loss_fn.eps_log))
                    t_sub = torch.log(t_sub.clamp(min=loss_fn.eps_log))

                sq_res = (p_sub - t_sub) ** 2
                norm_res = sq_res / (std_sub + eps) ** 2

                if hasattr(loss_fn, 'base_weight'):
                    with torch.no_grad():
                        tgt_filled = target.clone()
                        tgt_filled[torch.isnan(tgt_filled)] = 0.0
                        dq = torch.abs(tgt_filled[1:] - tgt_filled[:-1])
                        dq = torch.cat([dq[:1], dq], dim=0)
                        dq_max = dq.max(dim=0, keepdim=True).values.clamp(min=1e-6)
                        dq_norm = dq / dq_max
                        trend_w = dq_norm + loss_fn.base_weight
                    norm_res = norm_res * trend_w[mask_k]

                expert_loss = expert_loss + norm_res.mean()
            else:
                expert_loss = expert_loss + ((p_sub - t_sub) ** 2).mean()

        expert_loss = expert_loss / max(K, 1)

        # ---- Step 3: Quality floor (prevent non-winner degradation) ----
        qf_loss = torch.tensor(0.0, device=self.device)
        if self._mcl_quality_floor_weight > 0:
            # Unweighted per-expert NSE loss on ALL valid positions.
            obs = target.unsqueeze(0)  # [1, T, N]
            all_errors = (self._expert_streamflow - obs) ** 2  # [K, T, N]
            all_errors = all_errors.clamp(max=1e6)
            if self._basin_obs_std is not None and self._batch_sample is not None:
                std = self._basin_obs_std[self._batch_sample]
                std = std.unsqueeze(0).unsqueeze(0)
                all_errors = all_errors / (std + 0.1) ** 2
            all_errors[:, ~valid] = 0.0
            n_valid = valid.sum().clamp(min=1).float()
            qf_loss = all_errors.sum() / (K * n_valid)

        # ---- Step 4: Gate oracle loss ----
        gate_loss = torch.tensor(0.0, device=self.device)
        if self.gate_logits is not None:
            logits = self.gate_logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(0).expand(T_exp, -1, -1)
            elif logits.shape[0] > T_exp:
                logits = logits[-T_exp:]

            logits_flat = logits[valid]
            labels_flat = winners[valid]

            if logits_flat.numel() > 0:
                gate_loss = F.cross_entropy(logits_flat, labels_flat)

        # ---- Step 5: Optional main loss on gated combination ----
        main_loss = torch.tensor(0.0, device=self.device)
        if self._mcl_main_loss_weight > 0 and self.ensemble_predictions:
            pred_combined = self.ensemble_predictions.get('streamflow')
            if pred_combined is not None:
                main_loss = shared_loss_fn(
                    pred_combined.squeeze(),
                    target,
                    sample_ids=batch_sample,
                )

        loss = (
            self._mcl_expert_loss_weight * expert_loss
            + self._mcl_gate_loss_weight * gate_loss
            + self._mcl_quality_floor_weight * qf_loss
            + self._mcl_main_loss_weight * main_loss
        )
        return loss

    def train(self, mode: bool = True) -> 'MixtureOfExperts':
        """Set training mode; keep frozen experts in eval."""
        super().train(mode)
        if self.freeze_experts:
            for expert in self.experts.values():
                expert.eval()
        elif self.expert_lr:
            # Selective: unfrozen experts train, frozen ones stay eval.
            for name, expert in self.experts.items():
                if name in self._unfrozen_expert_names:
                    expert.train(mode)
                else:
                    expert.eval()
        return self
