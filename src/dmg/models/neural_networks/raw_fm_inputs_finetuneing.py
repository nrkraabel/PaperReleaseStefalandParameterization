"""No-foundation-model control for the embedding line of work.

EmbeddingFinetuneing asks what a pretrained foundation model's *output* buys
you. This model asks the control question: what do the foundation model's own
*inputs* buy you, with the encoder removed entirely?

The raw variables the MFFormer would have consumed
(``pretrained_time_series_vars`` / ``pretrained_static_vars``, read from
``pretrained_path`` by NnDualLoader) are handed to the adapter as additional
time/static inputs alongside the task's own forcings/attributes. Nothing is
encoded, nothing is pretrained, and no embedding file is read.

Everything downstream of the adapter -- adapter type, LSTM decoder, residual
path, projection -- is deliberately identical to EmbeddingFinetuneing, so a
delta against the daily-embedding baseline is attributable to the foundation
model rather than to a change of decoder.

The one structural difference is the adapter's residual base. In
EmbeddingFinetuneing the embedding *is* the hidden stream that the adapter
residually refines; here there is no such representation, so the hidden stream
is zeros and every FM variable arrives through the adapter's ordinary
time/static input ports. That is what "added as additional inputs" means: the
FM variables are conditioning features, never a hidden representation.

(A different control would linearly project the raw FM variables to d_model
and use that as the hidden stream -- i.e. replace the pretrained encoder with
a trivial one. That is a *weaker* ablation, since it reintroduces an encoder;
it is deliberately not what this model does.)
"""

import logging
from typing import Dict, Union

import torch
import torch.nn as nn
from models.neural_networks.adapters.build_adapter import apply_adapter, build_adapter
from models.neural_networks.cudnn_lstm import CudnnLstm
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class RawFmInputsFinetuneing(nn.Module):
    """Adapter + LSTM over task variables plus raw foundation-model inputs.

    ACCEPTS_BATCH_DICT = True signals NnModel to pass the full batch Dict
    rather than extracting xc_nn_norm, since this model also needs the raw
    pretrained-variable stream (carried under 'xc_pretrained_norm', the same
    key NnDualLoader already fills for DirectFinetuneing).
    """

    ACCEPTS_BATCH_DICT = True

    def __init__(self, config: Union[Dict, DictConfig], ny) -> None:
        super().__init__()

        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        if 'nn' in config_dict:
            nn_config = config_dict['nn']
        else:
            nn_config = config_dict.get('model', {}).get('nn', {})

        # Task-side variables (read from data_path).
        self.finetuning_ts_vars = nn_config.get('forcings', [])
        self.finetuning_static_vars = nn_config.get('attributes', [])
        if not self.finetuning_ts_vars or not self.finetuning_static_vars:
            raise ValueError("Must specify 'forcings' and 'attributes'")

        # Foundation-model-side variables (read from pretrained_path). Same two
        # config keys DirectFinetuneing uses to size its encoder -- here they
        # only say how to split xc_pretrained_norm back into its time and
        # static halves, since NnDualLoader concatenates them in that order.
        self.pretrained_ts_vars = nn_config.get('pretrained_time_series_vars', [])
        self.pretrained_static_vars = nn_config.get('pretrained_static_vars', [])
        if not self.pretrained_ts_vars and not self.pretrained_static_vars:
            raise ValueError(
                "RawFmInputsFinetuneing needs 'pretrained_time_series_vars' "
                "and/or 'pretrained_static_vars', plus a data_loader "
                "(NnDualLoader) that reads them from 'pretrained_path'. "
                "Without them this model is just the task-only baseline."
            )

        self.n_ft_ts = len(self.finetuning_ts_vars)
        self.n_ft_static = len(self.finetuning_static_vars)
        self.n_fm_ts = len(self.pretrained_ts_vars)
        self.n_fm_static = len(self.pretrained_static_vars)

        # Adapter/decoder width. No embedding is read, so this is purely the
        # internal width -- kept as 'embedding_size' so it can be set to the
        # same value as the embedding baseline for a capacity-matched compare.
        self.d_model = nn_config.get('embedding_size') or nn_config.get('hidden_size')
        if not self.d_model:
            raise ValueError("Must specify 'embedding_size' (or 'hidden_size').")

        self.dropout = nn_config.get('dropout', 0.1)
        self.use_residual_lstm = nn_config.get('use_residual_lstm', False)
        self.lstm_hidden_size = nn_config.get('lstm_hidden_size', self.d_model)
        self.target_variables = ny

        self.adapter_type = nn_config.get('adapter_type', 'dual_residual')
        adapter_params = nn_config.get('adapter_params', {})

        # Task vars + FM vars, everywhere the baseline used task vars alone.
        n_ts = self.n_ft_ts + self.n_fm_ts
        n_static = self.n_ft_static + self.n_fm_static

        self.adapter = build_adapter(
            self.adapter_type,
            self.d_model,
            n_ts,
            n_static,
            adapter_params,
        )

        self.decoder = CudnnLstm(
            nx=self.d_model,
            hidden_size=self.lstm_hidden_size,
            dr=self.dropout,
        )

        if self.use_residual_lstm:
            self.pre_lstm = nn.Linear(self.d_model + n_ts + n_static, self.d_model)
            self.post_lstm = nn.Linear(
                self.lstm_hidden_size + n_ts + n_static, self.lstm_hidden_size
            )

        self.projection = nn.Linear(self.lstm_hidden_size, self.target_variables)

        log.info(
            f"Initialized RawFmInputsFinetuneing: adapter={self.adapter_type}, "
            f"d_model={self.d_model} (no embedding is read)"
        )
        log.info(
            f"Adapter inputs: {n_ts} ts ({self.n_ft_ts} task + {self.n_fm_ts} FM), "
            f"{n_static} static ({self.n_ft_static} task + {self.n_fm_static} FM). "
            "Hidden/residual stream is zeros -- the foundation model is skipped."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, xc_nn_norm, temporal_features=None):
        """
        Parameters
        ----------
        xc_nn_norm : Dict
            From dpl_model/NnModel: 'xc_nn_norm' [T, B, n_ft_ts + n_ft_static]
            and 'xc_pretrained_norm' [T, B, n_fm_ts + n_fm_static], the raw
            (normalized, un-encoded) foundation-model variables.
        """
        fm = None
        if isinstance(xc_nn_norm, Dict):
            data_dict = xc_nn_norm
            xc_nn_norm = data_dict['xc_nn_norm']
            fm = data_dict.get('xc_pretrained_norm', None)

        if fm is None or fm.shape[-1] == 0:
            raise ValueError(
                "RawFmInputsFinetuneing requires the raw foundation-model "
                "variable stream under 'xc_pretrained_norm'. Use NnDualLoader "
                "as data_loader and set 'pretrained_path' in the config."
            )

        # Task features -> batch-first.
        batch_x_ft = xc_nn_norm[..., : self.n_ft_ts].permute(1, 0, 2)  # [B, T, n]
        batch_c_ft = xc_nn_norm[0, :, self.n_ft_ts : self.n_ft_ts + self.n_ft_static]

        fm = fm.permute(1, 0, 2)  # [T, B, F] -> [B, T, F]

        expected = self.n_fm_ts + self.n_fm_static
        if fm.shape[-1] != expected:
            raise ValueError(
                f"xc_pretrained_norm has width {fm.shape[-1]}, expected "
                f"{expected} (= {self.n_fm_ts} pretrained_time_series_vars + "
                f"{self.n_fm_static} pretrained_static_vars). The config's "
                "variable lists must match what the loader actually read."
            )
        # NnDualLoader aligns the pretrained stream onto the task's station and
        # calendar axes, so a mismatch here means that alignment did not happen
        # -- fail loudly rather than broadcast basins against each other.
        if fm.shape[0] != batch_x_ft.shape[0] or fm.shape[1] != batch_x_ft.shape[1]:
            raise ValueError(
                f"Foundation-model stream {tuple(fm.shape[:2])} does not match "
                f"task stream {tuple(batch_x_ft.shape[:2])} on (basin, time)."
            )

        # NnDualLoader concatenates [ts..., static...] and broadcasts the
        # statics across time, so t=0 recovers the per-basin static block.
        fm_ts = fm[..., : self.n_fm_ts]                      # [B, T, n_fm_ts]
        fm_static = fm[:, 0, self.n_fm_ts :]                 # [B, n_fm_static]

        time_features = torch.cat([batch_x_ft, fm_ts], dim=-1)
        static_features = torch.cat([batch_c_ft, fm_static], dim=-1)

        # No hidden representation exists -- the adapter refines zeros, so all
        # foundation-model information arrives as conditioning inputs only.
        hidden = batch_x_ft.new_zeros(
            batch_x_ft.shape[0], batch_x_ft.shape[1], self.d_model
        )

        adapted = apply_adapter(
            self.adapter,
            self.adapter_type,
            hidden,
            time_features,
            static_features,
        )

        if self.use_residual_lstm:
            static_exp = static_features.unsqueeze(1).expand(-1, adapted.size(1), -1)
            lstm_in = self.pre_lstm(
                torch.cat([adapted, time_features, static_exp], dim=-1)
            )
            lstm_in_t = lstm_in.permute(1, 0, 2)  # [T, B, d_model]
            lstm_out_t, _ = self.decoder(
                lstm_in_t, do_drop_mc=False, dr_false=(not self.training)
            )
            lstm_out = lstm_out_t.permute(1, 0, 2)  # [B, T, hidden]
            post = self.post_lstm(
                torch.cat([lstm_out, time_features, static_exp], dim=-1)
            )
            output = (post + lstm_out).permute(1, 0, 2)  # [T, B, hidden]
        else:
            adapted_t = adapted.permute(1, 0, 2)  # [T, B, d_model]
            output, _ = self.decoder(
                adapted_t, do_drop_mc=False, dr_false=(not self.training)
            )

        return self.projection(output)  # [T, B, ny]
