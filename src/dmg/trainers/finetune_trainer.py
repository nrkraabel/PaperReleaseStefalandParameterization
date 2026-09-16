"""
FinetuneTrainer
===============
Trainer for foundation-model fine-tuning experiments (DirectFinetuneing) and
physics-based differentiable model fine-tuning.

Extends BaseTrainer from the dmg package. Spatial testing is delegated to
dmg's built-in run_spatial_testing, so this class only handles per-epoch
training and per-split evaluation.

Key responsibilities retained here (not yet in the package):
  - HBV output dict unpacking (streamflow key)
  - AMP + ReduceLROnPlateau scheduler
  - Windowed evaluation for long sequences
"""

import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    """Trainer for fine-tuning and physics-augmented NN experiments."""

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.device = config['device']
        self.verbose = verbose

        self.is_in_train = False
        self.epoch_train_loss_list: list[float] = []
        self.epoch_val_loss_list: list[float] = []

        self.sampler = import_data_sampler(config['data_sampler'])(config)

        if 'train' in config['mode']:
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['train']['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func
            self.optimizer = optimizer or self.init_optimizer()
            self.start_epoch = config['train'].get('start_epoch', 0) + 1
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.get('lr_patience', 5),
                factor=config.get('lr_factor', 0.1),
            )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.

        Adding additional optimizers is possible by extending the optimizer_dict.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        name = self.config['train']['optimizer']['name']
        learning_rate = self.config['train']['lr']
        optimizer_dict = {
            # 'SGD': torch.optim.SGD,
            # 'Adam': torch.optim.Adam,
            # 'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            # 'RMSprop': torch.optim.RMSprop,
        }

        # Fetch optimizer class
        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(
                f"Optimizer '{name}' not recognized. "
                f"Available options are: {list(optimizer_dict.keys())}",
            )

        # Initialize
        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=learning_rate,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        epochs = self.config['train']['epochs']
        log.info(f"Training: epochs {self.start_epoch}-{epochs}")

        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = open(os.path.join(results_dir, 'results.txt'), 'a')
        results_file.write(f"\nTrain start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        use_amp = self.config.get('use_amp', False) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'], self.config
        )
        log.info(
            f"Training grid - basins: {n_samples}, minibatches: {n_minibatch}, timesteps: {n_timesteps}"
        )

        for epoch in range(self.start_epoch, epochs + 1):
            train_loss: list[float] = []
            epoch_time = time.time()
            self.model.train()

            for _ in range(n_minibatch):
                self.optimizer.zero_grad()
                batch = self.sampler.get_training_sample(
                    self.train_dataset, n_samples, n_timesteps
                )

                cm = torch.cuda.amp.autocast() if scaler else nullcontext()
                with cm:
                    # outputs = self.model(batch)
                    # target = batch['target']
                    _ = self.model(batch)
                    loss = self.model.calc_loss(batch)

                    # # Handle both raw-tensor models (DirectFinetuneing) and
                    # # dict-returning models (HBV wrappers)
                    # if torch.is_tensor(outputs):
                    #     model_output = outputs
                    # elif isinstance(outputs, dict):
                    #     raw = outputs.get('Hbv_1_1p', outputs)
                    #     model_output = (
                    #         raw['streamflow'] if isinstance(raw, dict) and 'streamflow' in raw
                    #         else raw
                    #     )
                    # else:
                    #     raise ValueError(f"Unexpected model output type: {type(outputs)}")

                    # sample_ids = batch.get('batch_sample')
                    # if sample_ids is None:
                    #     raise KeyError("batch_data missing 'batch_sample' (required by NseBatchLoss)")

                    # loss = self.loss_func(model_output, target, sample_ids=sample_ids)
                    if not torch.isnan(loss):
                        train_loss.append(loss.item())

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            self._log_epoch_stats(epoch, train_loss, [], epoch_time, results_file)

            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)

        results_file.close()
        log.info("Training complete")

    # ------------------------------------------------------------------
    # Evaluation dispatcher
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        self.is_in_train = False
        return self._evaluate_standard()

    # ------------------------------------------------------------------
    # Standard (streamflow / hydrology) evaluation
    # ------------------------------------------------------------------

    def _evaluate_standard(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        warmup = int((self.config.get('model') or {}).get('warmup', 0))
        rho = int((self.config.get('model') or {}).get('rho', 365))
        window_size = (
            warmup + rho
        )  # NnModel strips warmup from output, leaving rho steps

        obs = self.eval_dataset['target']
        obs_t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        if obs_t.ndim != 3:
            raise ValueError(
                f"Expected target ndim=3, got {obs_t.ndim} {tuple(obs_t.shape)}"
            )

        # Detect [N,T,1] vs [T,N,1]; raw target includes warmup rows at the start
        if obs_t.shape[1] >= obs_t.shape[0]:
            N_total, T_with_warmup = int(obs_t.shape[0]), int(obs_t.shape[1])
            targets_np = obs_t.detach().cpu().numpy()[:, warmup:, 0].transpose(1, 0)
        else:
            T_with_warmup, N_total = int(obs_t.shape[0]), int(obs_t.shape[1])
            targets_np = obs_t.detach().cpu().numpy()[warmup:, :, 0]

        T_eval = T_with_warmup - warmup  # actual prediction period

        batch_size = int(self.config.get('test', {}).get('batch_size', N_total))
        use_amp = bool(self.config.get('use_amp', False) and torch.cuda.is_available())
        model_name = ((self.config.get('model') or {}).get('nn') or {}).get(
            'name'
        ) or type(self.model).__name__

        # Window starts in T_with_warmup space; stride=rho, window=warmup+rho
        # NnModel strips warmup from model output -> rho predictions stored at pred[t0:t0+rho]
        last_start = max(0, T_with_warmup - window_size)
        t_starts = list(range(0, last_start + 1, rho))
        if not t_starts:
            t_starts = [0]
        if t_starts[-1] != last_start:
            t_starts.append(last_start)

        target_name = self.config['train']['target'][0]
        starts = np.arange(0, N_total, batch_size, dtype=int)
        ends = np.append(starts[1:], N_total).astype(int)
        batch_predictions: list[dict[str, torch.Tensor]] = []

        # eval_output_key may be a str or list; first entry is the primary key
        # (saved as Runoff.npy and used for NSE/KGE); extras are saved as {key}.npy
        # and fed to their eval scripts.
        _raw = self.config.get('eval_output_key', 'streamflow')
        _eval_keys: list[str] = [_raw] if isinstance(_raw, str) else list(_raw)
        _primary_hbv_key: str = _eval_keys[0]
        _aux_hbv_keys: list[str] = _eval_keys[1:]
        _aux_batch_preds: dict[str, list[torch.Tensor]] = {k: [] for k in _aux_hbv_keys}

        tf_full = self.eval_dataset.get('temporal_features')  # [T_tf, K] or None

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for s, e in zip(starts, ends):
                B = int(e - s)
                i_grid = np.arange(s, e)
                pred_be = torch.zeros((T_eval, B), device='cpu', dtype=torch.float32)
                _aux_bufs = {k: torch.zeros((T_eval, B), device='cpu', dtype=torch.float32)
                             for k in _aux_hbv_keys}

                for t0 in t_starts:
                    t1 = t0 + window_size
                    sample_w = {
                        'xc_nn_norm': self.eval_dataset['xc_nn_norm'][
                            t0:t1, i_grid, :
                        ].to(self.device),
                        'xc_pretrained_norm': self.eval_dataset['xc_pretrained_norm'][
                            t0:t1, i_grid, :
                        ].to(self.device),
                        'c_nn': self.eval_dataset['c_nn'][i_grid].to(self.device),
                        'c_phy': self.eval_dataset['c_phy'][i_grid].to(self.device),
                        'x_phy': self.eval_dataset['x_phy'][
                            t0:t1, i_grid, :
                        ].to(self.device),
                    }
                    if tf_full is not None:
                        t1_tf = min(t1, tf_full.shape[0])
                        sample_w['temporal_features'] = tf_full[t0:t1_tf].to(
                            self.device
                        )

                    out_dict = self.model(sample_w, eval=True)
                    if torch.is_tensor(out_dict):
                        out = out_dict
                    elif isinstance(out_dict, dict):
                        # out_dict is keyed by physics model name; model_name is the
                        # NN name, so fall back to the first value when key not found.
                        candidate = out_dict.get(model_name) or next(
                            iter(out_dict.values())
                        )
                        if torch.is_tensor(candidate):
                            out = candidate
                        elif isinstance(candidate, dict):
                            # Resolve primary key. Physics models (e.g. HBV) key their
                            # output by canonical hydrology names (streamflow, etc.),
                            # configurable via eval_output_key. NnModel (no physics,
                            # e.g. NoHBV configs) instead keys its output directly by
                            # the configured target name (e.g. 'QObs'), so that's
                            # tried too before giving up.
                            _pk = next(
                                (
                                    k
                                    for k in (_primary_hbv_key, 'streamflow', target_name)
                                    if candidate.get(k) is not None
                                ),
                                None,
                            )
                            if _pk is None:
                                raise ValueError(
                                    f"Cannot extract '{_primary_hbv_key}', 'streamflow', "
                                    f"or '{target_name}'. Non-None keys: "
                                    f"{[k for k, v in candidate.items() if v is not None]}"
                                )
                            out = candidate[_pk]
                            # Extract auxiliary keys into their own buffers.
                            for _ak in _aux_hbv_keys:
                                _at = candidate.get(_ak)
                                if _at is not None:
                                    if _at.ndim == 3 and _at.shape[2] == 1:
                                        _at = _at[:, :, 0]
                                    if _at.shape == (B, rho):
                                        _at = _at.transpose(0, 1)
                                    if _at.shape == (rho, B):
                                        _aux_bufs[_ak][t0:t0 + rho] = _at.detach().cpu()
                        else:
                            raise ValueError(
                                f"Cannot extract prediction tensor from model output: "
                                f"{list(out_dict.keys())}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected model output type: {type(out_dict)}"
                        )

                    if out.ndim == 3 and out.shape[2] == 1:
                        out = out[:, :, 0]
                    # normalise to [rho, B]
                    if out.shape == (B, rho):
                        out = out.transpose(0, 1)
                    elif out.shape != (rho, B):
                        raise ValueError(
                            f"Eval window shape {tuple(out.shape)}, expected ({rho},{B})"
                        )

                    pred_be[t0 : t0 + rho] = out.detach().cpu()

                # Store as [T_eval, B, 1] -- matches save_outputs convention
                batch_predictions.append({target_name: pred_be.unsqueeze(-1)})
                for _ak in _aux_hbv_keys:
                    _aux_batch_preds[_ak].append(_aux_bufs[_ak].unsqueeze(-1))

        # Delegate I/O to save_outputs (same pattern as Trainer.evaluate)
        sim_dir = self.config.get('sim_dir', self.config.get('out_path', 'results'))
        os.makedirs(sim_dir, exist_ok=True)
        cfg = {**self.config, 'sim_dir': sim_dir}
        obs_save = torch.from_numpy(targets_np[:, :, np.newaxis])  # [T_eval, N, 1]
        save_outputs(cfg, batch_predictions, obs_save)
        log.info(f"Saved eval outputs to: {sim_dir}")

        # Save each auxiliary HBV key as its own .npy and run its eval script.
        for _ak, _tensors in _aux_batch_preds.items():
            if _tensors:
                _cat = torch.cat(_tensors, dim=1)  # [T_eval, N, 1]
                _path = os.path.join(sim_dir, f'{_ak}.npy')
                np.save(_path, _cat.numpy())
                log.info(f"Saved {_ak} to: {_path}")
        if _aux_hbv_keys and self.config.get('data_path'):
            for _ak in _aux_hbv_keys:
                self._run_auxiliary_eval(_ak, sim_dir)

        pred = torch.cat([d[target_name] for d in batch_predictions], dim=1)[
            :, :, 0
        ].numpy()
        self.predictions = {target_name: pred}
        self.calc_metrics(batch_predictions, obs_save)
        return pred, obs

    def _run_auxiliary_eval(self, hbv_key: str, sim_dir: str) -> None:
        """Run eval_sm_recharge.py or eval_gw_percolation.py for an auxiliary HBV key."""
        _script_map = {
            'recharge':    'eval_sm_recharge.py',
            'percolation': 'eval_gw_percolation.py',
        }
        script_name = _script_map.get(hbv_key)
        if not script_name:
            return

        # Scripts live two directories up from this file: src/dmg/trainers -> scripts/
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '..', 'scripts',
        )
        script_path = os.path.abspath(os.path.join(scripts_dir, script_name))
        if not os.path.exists(script_path):
            log.warning(f"Auxiliary eval script not found: {script_path}")
            return

        nc_file = self.config.get('data_path', '')
        test_cfg = self.config.get('test', {})
        test_start = str(test_cfg.get('start_time', '1992-01-01')).replace('/', '-')
        test_end   = str(test_cfg.get('end_time',   '2018-12-30')).replace('/', '-')
        out_csv    = os.path.join(sim_dir, f'{hbv_key}_metrics.csv')

        cmd = [
            sys.executable, script_path,
            '--pred_dir',   sim_dir,
            '--pred_key',   hbv_key,
            '--nc_file',    nc_file,
            '--out_csv',    out_csv,
            '--test_start', test_start,
            '--test_end',   test_end,
        ]
        log.info(f"Running auxiliary eval: {script_name}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for line in result.stdout.strip().splitlines():
                log.info(f"[{script_name}] {line}")
            if result.returncode != 0:
                log.warning(f"Auxiliary eval failed:\n{result.stderr.strip()[-500:]}")
        except subprocess.TimeoutExpired:
            log.warning(f"Auxiliary eval timed out (600 s)")
        except Exception as exc:
            log.warning(f"Auxiliary eval error: {exc}")

    # ------------------------------------------------------------------
    # BaseTrainer abstract method implementations
    # ------------------------------------------------------------------

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        target_name = self.config['train']['target'][0]
        predictions = (
            torch.cat([d[target_name] for d in batch_predictions], dim=1).cpu().numpy()
        )
        target = np.expand_dims(observations[..., 0].cpu().numpy(), 2)

        # Apply denormalization if the loader stored a denorm function
        denorm_fn = self.eval_dataset.get('denorm_fn') if self.eval_dataset else None
        if denorm_fn is not None:
            predictions = denorm_fn(predictions)

        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )
        sim_dir = self.config.get('sim_dir', self.config.get('out_path', 'results'))
        metrics.dump_metrics(sim_dir)

    def inference(self) -> None:
        raise NotImplementedError("Inference is not implemented for FinetuneTrainer.")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_epoch_stats(
        self, epoch, train_loss, val_loss, epoch_time, results_file
    ) -> None:
        train_avg = float(np.mean(train_loss)) if train_loss else float('nan')
        val_avg = float(np.mean(val_loss)) if val_loss else None

        self.epoch_train_loss_list.append(train_avg)
        if val_avg is not None:
            self.epoch_val_loss_list.append(val_avg)

        msg = (
            f"Epoch {epoch}: train_loss={train_avg:.4f}"
            + (f", val_loss={val_avg:.4f}" if val_avg is not None else "")
            + f" ({time.time() - epoch_time:.1f}s)"
        )
        log.info(msg)
        results_file.write(msg + '\n')
        results_file.flush()
        self._save_loss_data(epoch, train_avg, val_avg)

    def _save_loss_data(
        self, epoch: int, train_loss: float, val_loss: Optional[float]
    ) -> None:
        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'loss_data.csv'), 'a') as f:
            f.write(f"{epoch},{train_loss},{'' if val_loss is None else val_loss}\n")


# ---------------------------------------------------------------------------
# Module-level helpers (avoid polluting the class namespace)
# ---------------------------------------------------------------------------


def _slice_time(sample: dict, T_total: int, t0: int, t1: int, device: str) -> dict:
    """Slice the time dimension of all tensors in a batch dict and move to device."""
    out = {}
    for k, v in sample.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if v.ndim == 3:
            if v.shape[0] == T_total:
                out[k] = v[t0:t1].to(device)
            elif v.shape[1] == T_total:
                out[k] = v[:, t0:t1].to(device)
            else:
                out[k] = v.to(device)
        elif v.ndim == 2 and v.shape[0] == T_total:
            out[k] = v[t0:t1].to(device)
        else:
            out[k] = v.to(device)
    return out
