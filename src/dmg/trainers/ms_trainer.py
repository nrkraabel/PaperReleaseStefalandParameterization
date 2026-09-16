import gc
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from torch.amp import GradScaler

from dmg.core.calc.metrics import Metrics
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class _BatchPrefetch:
    """Prefetch training batches in a background thread.

    NOTE: Runs sampler one batch ahead so that CPU-bound
    numpy work (array allocation, slicing, axis swaps) overlaps with GPU
    forward/backward passes.

    Parameters
    ----------
    sampler
        Data sampler with a get_training_sample method.
    dataset
        Training dataset dictionary passed to the sampler.
    n_samples
        Number of gages in the training set.
    nt
        Number of timesteps in the training set.
    n_batches
        Total number of batches to prefetch.
    """

    def __init__(self, sampler, dataset, n_samples, nt, n_batches):
        self._sampler = sampler
        self._dataset = dataset
        self._n_samples = n_samples
        self._nt = nt
        self._n_batches = n_batches
        self._queue = queue.Queue(maxsize=1)

    def _worker(self):
        try:
            for _ in range(self._n_batches):
                # Build batch on CPU to avoid fighting for vram.
                batch = self._sampler.get_training_sample(
                    self._dataset,
                    self._n_samples,
                    self._nt,
                    device='cpu',
                )
                # Pin memory
                batch = {
                    k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                self._queue.put(batch)
        except RuntimeError as exc:
            self._queue.put(exc)

    def __iter__(self):
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()
        for _ in range(self._n_batches):
            item = self._queue.get()
            if isinstance(item, RuntimeError):
                raise item
            yield item
        thread.join()


class MsTrainer(BaseTrainer):
    """Multiscale trainer for differentiable models.

    Handles training, evaluation, and inference for multiscale models that
    operate on catchment-scale data aggregated to gage outlets.

    Follows the same structure as the standard Trainer but adapted for
    multiscale batch construction via the MsHydroSampler. Training uses
    mixed-precision (AMP) by default on CUDA devices and the Adadelta
    optimizer, matching the original multiscale training pipeline.

    Parameters
    ----------
    config
        Configuration settings for the model and experiment.
    model
        Learnable model object. If not provided, a new model is initialized.
    train_dataset
        Training dataset dictionary containing gage-level and catchment-level data.
    eval_dataset
        Testing/evaluation dataset dictionary.
    dataset
        Inference dataset dictionary.
    loss_func
        Loss function object. If not provided, a new loss function is
        initialized from config.
    optimizer
        Optimizer object. If not provided, a new optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is
        initialized from config.
    write_out
        Whether to save model outputs and metrics to disk.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        write_out: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config, verbose=verbose)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.write_out = write_out
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False
        self.exp_logger = None

        if 'train' in config['mode']:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            log.info("Initializing multiscale training experiment")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['train']['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config['train']['lr_scheduler']:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            self.start_epoch = self.config['train']['start_epoch'] + 1
            if self.start_epoch > 1:
                self.load_states()

            # Mixed precision scaler for accelerating training with cuda.
            self.use_amp = config['train'].get(
                'use_amp',
                config['device'] != 'cpu',
            )
            self.scaler = GradScaler() if self.use_amp else None

            self._init_loggers()
            self._init_loss_tracking()

    def _init_loss_tracking(self) -> None:
        """Initialize loss history lists and CSV log file."""
        self.train_loss_history: list[float] = []
        self.loss_component_history: dict[str, list[float]] = {}

        if self.write_out:
            self.plot_dir = self.config['plot_dir']

            self.csv_log_file = os.path.join(
                self.config['output_dir'], 'training_log.csv'
            )
            with open(self.csv_log_file, 'w') as f:
                f.write('epoch,batch,loss,time_s,gpu_mem_mb\n')

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.

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
                self.model.get_parameters(), lr=learning_rate, foreach=True
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler for the optimizer.

        Returns
        -------
        torch.optim.lr_scheduler.LRScheduler
            Initialized learning rate scheduler object.
        """
        params = self.config['train']['lr_scheduler'].copy()
        name = params.pop('name')
        scheduler_dict = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            # 'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. "
                f"Available options are: {list(scheduler_dict.keys())}",
            )

        try:
            self.scheduler = cls(self.optimizer, **params)
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.
        """
        path = self.config['model_dir']
        target_epoch = self.start_epoch - 1

        for file in os.listdir(path):
            if ('trainer_state' in file) and (f'ep{target_epoch}' in file):
                log.info(
                    f"Loading trainer states -> Resuming from epoch {self.start_epoch}",
                )

                checkpoint = torch.load(os.path.join(path, file))

                # Restore optimizer states
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'],
                )
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(
                        checkpoint['scheduler_state_dict'],
                    )

                # Restore random states
                torch.set_rng_state(checkpoint['random_state'])
                if torch.cuda.is_available() and checkpoint.get('cuda_state'):
                    torch.cuda.set_rng_state(checkpoint['cuda_state'])
                if checkpoint.get('numpy_random_state') is not None:
                    np.random.set_state(checkpoint['numpy_random_state'])
                return
            elif 'train_state' in file:
                raise FileNotFoundError(
                    f"Available checkpoint {file} does"
                    f" not match start epoch {self.start_epoch - 1}.",
                )

        log.warning(
            f"No checkpoint found for epoch {target_epoch}. Starting from first epoch.",
        )

    def train(self) -> None:
        """Entrypoint for the multiscale training loop."""
        self.is_in_train = True

        # Setup a training grid (number of samples, minibatches, and timesteps)
        nt, n_samples, _ = self.train_dataset['target'].shape

        # Calculate iter/ep
        rho = self.config['model']['rho']
        warmup = self.config['model'].get('warmup', 0)
        batch_size = self.config['train']['batch_size']
        n_minibatch = int(
            np.ceil(
                np.log(0.01)
                / np.log(
                    1 - batch_size * rho / n_samples / (nt - warmup),
                ),
            ),
        )

        log.info(
            f"Training model: {self.start_epoch} to {self.epochs} epochs, "
            f"{n_minibatch} iter/ep, {n_samples} gages",
        )

        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Disable garbage collection during training for performance
            gc.collect()
            gc.disable()

            self._train_one_epoch(
                epoch,
                n_samples,
                n_minibatch,
                nt,
            )

            # Re-enable garbage collection
            gc.enable()
            gc.collect()

        self.exp_logger.finalize()

    def _plot_loss_curves(self) -> None:
        """Generate and save training loss plots (linear and log scale)."""
        if not self.train_loss_history:
            return

        epochs = range(1, len(self.train_loss_history) + 1)
        save_path = Path(self.plot_dir) / 'loss_plot.png'

        multi_model = len(self.loss_component_history) > 1

        for log_scale, suffix in [(False, ''), (True, '_log')]:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                self.train_loss_history,
                label='Total Loss' if multi_model else None,
                color='blue',
                linewidth=1.5,
            )

            if multi_model:
                colors = ['orange', 'green', 'red', 'purple', 'brown']
                for i, (name, losses) in enumerate(self.loss_component_history.items()):
                    ax.plot(
                        epochs,
                        losses,
                        label=name,
                        color=colors[i % len(colors)],
                        linewidth=1.5,
                        linestyle='--',
                    )

            title = 'Training Loss'
            if log_scale:
                ax.set_yscale('log')
                title += ' (Log Scale)'
                ax.grid(True, which='both', ls='--', linewidth=0.5, alpha=0.7)
            else:
                ax.grid(True, ls='--', linewidth=0.5, alpha=0.7)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            if multi_model:
                ax.legend(loc='upper right', fontsize=10)
            fig.tight_layout()

            out = save_path.with_stem(f"{save_path.stem}{suffix}")
            fig.savefig(out, dpi=150)
            plt.close(fig)

    def _train_one_epoch(
        self,
        epoch: int,
        n_samples: int,
        n_minibatch: int,
        nt: int,
    ) -> None:
        """Train model for one epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        n_sampless
            Number of samples in the training dataset.
        n_minibatch
            Number of minibatches (iterations) per epoch.
        nt
            Number of timesteps in the training dataset.
        """
        start_time = time.perf_counter()

        self.current_epoch = epoch
        self.total_loss = 0.0

        if hasattr(self.model, 'loss_dict'):
            for key in self.model.loss_dict:
                self.model.loss_dict[key] = 0.0

        prog_bar = tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        # Prefetch batches (synchronous cpu sampling + async cpu-gpu transfer)
        prefetcher = _BatchPrefetch(
            self.sampler,
            self.train_dataset,
            n_samples,
            nt,
            n_minibatch,
        )
        device = self.config['device']

        # Iterate through epoch in minibatches
        for mb, dataset_sample in zip(prog_bar, prefetcher):
            self.current_batch = mb
            batch_start = time.perf_counter()

            # Async batch transfer (pinned mem for non-blocking).
            dataset_sample = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in dataset_sample.items()
            }

            # Forward pass (+ mixed prec)
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = self.model(dataset_sample)
                    self._aggregate_to_gage(dataset_sample)
                self._free_tensors()
                self._strip_warmup()
                loss = self.model.calc_loss(dataset_sample)
            else:
                _ = self.model(dataset_sample)
                self._aggregate_to_gage(dataset_sample)
                self._free_tensors()
                self._strip_warmup()
                loss = self.model.calc_loss(dataset_sample)

            # Skip nan loss (avoid model weight corruption)
            if torch.isnan(loss):
                log.warning(
                    f"Epoch {epoch}, batch {mb}: NaN loss -- skipping update.",
                )
                self.optimizer.zero_grad()
                # torch.cuda.empty_cache()
                continue

            # Backward pass (+ grad scaler for mixed prec)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()
            batch_loss = loss.item()
            self.total_loss += batch_loss
            # torch.cuda.empty_cache()  # Disabled: causes ~1s/batch overhead!

            print(f"Epoch {epoch}, Batch {mb}, Loss: {batch_loss:.7f}")

            if self.write_out:
                batch_elapsed = time.perf_counter() - batch_start
                mem = 0
                if self.config['device'] != 'cpu':
                    mem = int(
                        torch.cuda.memory_reserved(device=self.config['device']) * 1e-6
                    )
                with open(self.csv_log_file, 'a') as f:
                    f.write(
                        f"{epoch},{mb},{batch_loss:.6f},{batch_elapsed:.2f},{mem}\n"
                    )

        if self.use_scheduler:
            self.scheduler.step()

        if self.verbose:
            log.info(f"\n ---- \n Epoch {epoch} total loss: {self.total_loss}")
        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

        # Save model and trainer states
        if (epoch % self.config['train']['save_epoch'] == 0) and self.write_out:
            self.model.save_model(epoch)
            save_train_state(
                self.config['model_dir'],
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )

    def evaluate(self) -> None:
        """Run gage-level model eval with cat-gage aggregation.

        Groups gages into batches whose total cat count does not exceed batch
        size. For each batch the model produces cat-level output which is then
        area-weighted and aggregated to gage-level before computing metrics.
        """
        self.is_in_train = False

        observations = self.eval_dataset['target']
        warmup = self.config['model'].get('warmup', 0)
        batch_size = self.config['test']['batch_size']

        gage_key = self.eval_dataset['gage_key']
        cat_idx = self.eval_dataset['cat_idx']
        n_gages = len(gage_key)

        # Accumulate gages until the total cat count == batch_size
        cum_cat = 0
        ncat_list = []
        for gage in gage_key:
            cum_cat += len(cat_idx[gage])
            ncat_list.append(cum_cat)

        iS = [0]
        prev_ncat = 0
        for i, nc in enumerate(ncat_list):
            if i > 0 and (nc - prev_ncat) >= batch_size:
                iS.append(i)
                prev_ncat = ncat_list[i - 1]

        iS = np.array(iS)
        iE = np.append(iS[1:], n_gages)

        # Model forward with cat-gage aggregation
        batch_predictions = []
        log.info(
            f"Evaluating Model: {len(iS)} batches, {n_gages} gages",
        )

        prog_bar = tqdm.tqdm(
            range(len(iS)),
            desc='Evaluating',
            leave=False,
            dynamic_ncols=True,
        )

        for i in prog_bar:
            self.current_batch = i

            dataset_sample = self.sampler.get_eval_sample(
                self.eval_dataset,
                iS[i],
                iE[i],
            )

            # Forward pass for cat-level output
            self.model(dataset_sample, eval=True)

            # Aggregate cat to gage
            self._aggregate_to_gage(dataset_sample)

            # Extract gage-level preds and strip warmup
            model_name = self.config['model']['phy']['name'][0]
            prediction = {
                key: tensor[warmup:].detach().cpu() if tensor is not None else None
                for key, tensor in self.model.output_dict[model_name].items()
            }
            batch_predictions.append(prediction)
            # torch.cuda.empty_cache()

        # Save preds
        log.info("Saving model outputs + Calculating metrics")
        save_outputs(self.config, batch_predictions, observations)
        self.predictions = self._batch_data(batch_predictions)

        # Calculate metrics
        self.calc_metrics(batch_predictions, observations)
        torch.cuda.empty_cache()

    def inference(self) -> None:
        """Run batch model inference at cat-level and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch.
        n_samples = self.dataset['x_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['sim']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Forward loop
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.

        Parameters
        ----------
        batch_list
            List of dictionaries containing batch data.
        target_key
            Key to extract from each batch dictionary.
        """
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if batch_list[0][key] is None:
                    data[key] = None
                    continue
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = (
                    torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
                )
            return data

        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ) -> None:
        """Forward loop used in model evaluation and inference.

        Parameters
        ----------
        data
            dictionary containing model input data.
        batch_start
            Start indices for each batch.
        batch_end
            End indices for each batch.
        """
        # Track predictions accross batches
        batch_predictions = []

        prog_bar = tqdm.tqdm(
            range(len(batch_start)),
            desc='Forwarding',
            leave=False,
            dynamic_ncols=True,
        )

        for i in prog_bar:
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch preds
            model_name = self.config['model']['phy']['name'][0]
            prediction = {
                key: tensor.detach().cpu() if tensor is not None else None
                for key, tensor in prediction[model_name].items()
            }
            batch_predictions.append(prediction)
            # torch.cuda.empty_cache()
        return batch_predictions

    def _aggregate_to_gage(
        self,
        dataset_sample: dict[str, torch.Tensor],
    ) -> None:
        """Aggregate catchment-level model outputs to gage-level.

        Uses area-weighted outlet topology to convert cat-level predictions to
        gage-level:
                gage_q = (merit_q * areas) @ topo

        NOTE: During training, only the target variable(s) are aggregated to
        save vram.

        Parameters
        ----------
        dataset_sample
            Training batch dictionary containing areas and
            outlet_topo keys.
        """
        if 'outlet_topo' not in dataset_sample:
            return

        areas = dataset_sample['areas']
        outlet_topo = dataset_sample['outlet_topo']
        n_merit = areas.shape[0]

        target_keys = None
        if self.is_in_train:
            target_keys = set(self.config['train']['target'])

        for name in self.model.output_dict:
            for key, pred in list(self.model.output_dict[name].items()):
                if target_keys and key not in target_keys:
                    continue
                if pred is not None and pred.ndim == 3 and pred.shape[1] == n_merit:
                    pred_2d = pred.squeeze(-1)
                    agg = (pred_2d * areas.unsqueeze(0)) @ outlet_topo
                    self.model.output_dict[name][key] = agg.unsqueeze(-1)

    def _free_tensors(self) -> None:
        """
        Drop non-target flux variables from output_dict and clear model state
        cache to free gpu vram.
        """
        if not self.is_in_train:
            return

        target_keys = set(self.config['train']['target'])

        for name in self.model.output_dict:
            keys_to_drop = [
                k for k in self.model.output_dict[name] if k not in target_keys
            ]
            for k in keys_to_drop:
                del self.model.output_dict[name][k]

        # Clear cached state timeseries
        for _, model in self.model.model_dict.items():
            if hasattr(model, '_state_cache'):
                model._state_cache = None

    def _strip_warmup(self) -> None:
        """Strip warmup period from model predictions to align with target."""
        warmup = self.config['model'].get('warmup', 0)
        if warmup > 0:
            for name in self.model.output_dict:
                for key in self.model.output_dict[name]:
                    tensor = self.model.output_dict[name][key]
                    if tensor is not None:
                        self.model.output_dict[name][key] = tensor[warmup:]

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        target_name = self.config['train']['target'][0]
        warmup = self.config['model'].get('warmup', 0)
        predictions = self._batch_data(batch_predictions, target_name)

        if torch.is_tensor(observations):
            target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        else:
            target = np.expand_dims(observations[:, :, 0], 2)

        target = target[warmup:, :]

        # Compute metrics
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config['output_dir'])

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log statistics after each epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        loss_dict
            dictionary containing loss values.
        n_minibatch
            Number of minibatches.
        start_time
            Start time of the epoch.
        """
        avg_loss_dict = {key: value / n_minibatch for key, value in loss_dict.items()}
        avg_total_loss = self.total_loss / n_minibatch
        loss_str = ", ".join(
            f"{key}: {value:.6f}" for key, value in avg_loss_dict.items()
        )
        elapsed = time.perf_counter() - start_time
        mem_aloc = 0

        if self.config['device'] != 'cpu':
            mem_aloc = int(
                torch.cuda.memory_reserved(device=self.config['device']) * 1e-6,
            )

        log.debug(
            f"Loss after epoch {epoch}: {loss_str} \n"
            f"~ Runtime {elapsed:.2f} s, {mem_aloc} Mb reserved GPU memory",
        )

        # Track loss history
        self.train_loss_history.append(avg_total_loss)
        for model_name, loss_val in avg_loss_dict.items():
            if model_name not in self.loss_component_history:
                self.loss_component_history[model_name] = []
            self.loss_component_history[model_name].append(loss_val)

        # For experiment loggers: create a single dictionary of metrics to log
        metrics_to_log = {
            'Loss/train_total': avg_total_loss,
        }
        for model_name, loss_val in avg_loss_dict.items():
            metrics_to_log[f'Loss/{model_name}'] = loss_val

        if self.use_scheduler:
            metrics_to_log['learning_rate'] = self.scheduler.get_last_lr()[0]

        # Loop through all active loggers and log the metrics
        self.exp_logger.log_metrics(metrics_to_log, step=epoch)

        # Update loss plots
        if self.write_out:
            self._plot_loss_curves()
