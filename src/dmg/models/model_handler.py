import logging
import os
from typing import Any, Optional

import torch

from dmg.core.utils import save_model
from dmg.models.delta_models.dpl_model import DplModel
from dmg.models.multimodels.moe import MixtureOfExperts
from dmg.models.wrappers.nn_model import NnModel

log = logging.getLogger('model_handler')


class ModelHandler(torch.nn.Module):
    """Streamlines handling of differentiable models and multimodel ensembles.

    This interface additionally acts as a link to the CSDMS BMI, enabling
    compatibility with the NOAA-OWP NextGen framework.

    Features
    - Model initialization (new or from a checkpoint)
    - Loss calculation
    - Forwarding for single/multi-model setups
    - (Planned) Multimodel ensembles/loss and multi-GPU compute

    Parameters
    ----------
    config
        Configuration settings for the model.
    device
        Device to run the model on.
    verbose
        Whether to print verbose output.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = None,
        verbose=False,
    ) -> None:
        super().__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self.model_type = None
        self.model_path = config['model_dir']
        self.verbose = verbose

        if device is None:
            self.device = config['device']
        else:
            self.device = device

        self.multimodel_type = config['multimodel_type']
        self.model_dict = {}
        # TODO: add proper support for multiple targets...
        self.target_names = config['train']['target']

        # Maps dataset/config target variable names to physics model output keys.
        # Needed when a dataset uses different naming (e.g. 'Runoff') from the
        # model's standard output key (e.g. 'streamflow').
        self._target_output_aliases: dict[str, str] = {
            'Runoff': 'streamflow',
            'runoff': 'streamflow',
            'QObs': 'streamflow',
            'qobs': 'streamflow',
        }

        # MoE: experts are loaded internally; skip normal model init.
        if self.multimodel_type == 'moe':
            self.moe = MixtureOfExperts(
                moe_config=config['moe'],
                model_config=config['model'],
                device=self.device,
                model_dir=config.get('model_dir'),
            )
            self.models = ['MoE']

            # Load gate checkpoint for test/eval modes.
            if config['mode'] in ['test', 'sim']:
                epoch = config['test']['test_epoch']
                gate_path = os.path.join(
                    self.model_path,
                    f"moe_gate_ep{epoch}.pt",
                )
                if os.path.exists(gate_path):
                    self.moe.gate.load_state_dict(
                        torch.load(
                            gate_path, weights_only=True, map_location=self.device
                        ),
                    )
                    log.info(f"Loaded MoE gate from ep{epoch}")
            elif config['mode'] == 'train' and config['train']['start_epoch'] > 0:
                epoch = config['train']['start_epoch']
                gate_path = os.path.join(
                    self.model_path,
                    f"moe_gate_ep{epoch}.pt",
                )
                if os.path.exists(gate_path):
                    self.moe.gate.load_state_dict(
                        torch.load(
                            gate_path, weights_only=True, map_location=self.device
                        ),
                    )
                    log.info(f"Resumed MoE gate from ep{epoch}")
        else:
            self.models = self.list_models()
            self._init_models()

            if 'train' not in config['mode']:
                if config.get('load_state_path'):
                    self.load_states(config['load_state_path'])

        self.epoch = None
        self.loss_func = None
        self.loss_dict = dict.fromkeys(self.models, 0)

    def list_models(self) -> list[str]:
        """List of models specified in the configuration.

        TODO: Support physics-only forward.

        Returns
        -------
        list[str]
            List of model names.
        """
        if self.config['model']['phy']:
            models = self.config['model']['phy']['name']
            self.model_type = 'dm'
        elif self.config['model']['nn']:
            models = self.config['model']['nn']['name']
            self.model_type = 'nn'
        else:
            raise ValueError("No models specified in configuration.")

        if not isinstance(models, list):
            models = [models]

        return models

    def _init_models(self) -> None:
        """Initialize and store models, multimodels, and checkpoints."""
        if (self.multimodel_type is None) and (len(self.models) > 1):
            raise ValueError(
                "Multiple models specified, but ensemble type is 'none'. Check configuration.",
            )

        # Epoch to load
        if self.config['mode'] == 'train':
            load_epoch = self.config['train']['start_epoch']
        elif self.config['mode'] in ['test', 'sim']:
            load_epoch = self.config['test']['test_epoch']
        else:
            load_epoch = self.config.get('load_epoch', 0)

        # Load models
        try:
            self.load_model(load_epoch)
        except Exception as e:
            raise e

    def load_model(self, epoch: int = 0) -> None:
        """Load a specific model from a checkpoint.

        Parameters
        ----------
        epoch
            Epoch to load the model from.
        """
        for name in self.models:
            if self.model_type == 'nn':
                # Standalone neural network model
                self.model_dict[name] = NnModel(
                    target_names=self.target_names,
                    config=self.config['model'],
                    device=self.device,
                )
            else:
                # Differentiable model (dPL modality)
                # TODO: make dynamic import for other modalities.
                compile_nn = self.config.get('compile_nn', False)
                max_cat_size = self.config['train'].get('max_cat_size')
                self.model_dict[name] = DplModel(
                    phy_model_name=name,
                    config=self.config['model'],
                    device=self.device,
                    compile_nn=compile_nn,
                    nn_pad_size=max_cat_size if compile_nn else None,
                )

            if epoch == 0:
                self.epoch = 0

                # Leave model uninitialized for training.
                if self.verbose:
                    log.info(f"Created new model: {name}")
                continue
            else:
                self.epoch = epoch

                # Initialize model from checkpoint state dict.
                path = self.model_path
                if f"{name.lower()}_ep" not in path:
                    path = os.path.join(path, f"{name.lower()}_ep{epoch}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{path} not found for model {name}.",
                    )
                self.model_dict[name].load_state_dict(
                    torch.load(
                        path,
                        weights_only=True,
                        map_location=self.device,
                    ),
                    strict=False,
                )
                self.model_dict[name].to(self.device)

                # Overwrite internal config if there is discontinuity:
                if (self.model_type == 'dm') and self.model_dict[name].config:
                    self.model_dict[name].config = self.config['model']

                if self.verbose:
                    log.info(f"Loaded model: {name}, Ep {epoch}")

    def train(self, mode: bool = True) -> 'ModelHandler':
        """Set all models to training mode (or eval mode if mode=False).

        Overrides torch.nn.Module.train to propagate to models stored
        in model_dict (a plain dict, not a Moduledict).
        Since nn.Module.eval() delegates to train(False), this
        override covers both .train() and .eval() calls.

        Parameters
        ----------
        mode
            Whether to set training mode (True) or eval mode (False).

        Returns
        -------
        ModelHandler
            Self.
        """
        super().train(mode)
        if self.multimodel_type == 'moe':
            self.moe.train(mode)  # Gate trains; experts stay frozen/eval.
            return self
        for model in self.model_dict.values():
            model.train(mode)
        return self

    def get_parameters(self) -> list[torch.Tensor]:
        """Return all model parameters.

        Returns
        -------
        list[torch.Tensor]
            List of model parameters.
        """
        self.parameters = []

        if self.multimodel_type == 'moe':
            # Gate (and spatial prior) parameters.
            gate_params = list(self.moe.gate.parameters())
            if self.moe.spatial_prior_mlp is not None:
                gate_params += list(self.moe.spatial_prior_mlp.parameters())

            # AWL learned sigma parameters (trained at gate LR).
            if self.moe.awl is not None:
                gate_params += list(self.moe.awl.parameters())

            # If experts are unfrozen, include their parameters.
            if self.moe._unfrozen_expert_names:
                expert_params = []
                for name in self.moe._unfrozen_expert_names:
                    expert_params += list(
                        self.moe.experts[name].parameters(),
                    )
                # Store flat list for gradient clipping.
                self._flat_parameters = gate_params + expert_params

                if self.moe.expert_lr:
                    # Separate LR for experts (fine-tuning mode).
                    self.parameters = [
                        {'params': gate_params},
                        {'params': expert_params, 'lr': self.moe.expert_lr},
                    ]
                else:
                    # All at main LR (from-scratch training).
                    self.parameters = gate_params + expert_params
            else:
                self._flat_parameters = gate_params
                self.parameters = gate_params
            return self.parameters

        for model in self.model_dict.values():
            self.parameters += list(model.parameters())
        return self.parameters

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Sequentially forward one or more models with an optional weighting NN
        for multimodel ensembles trained in parallel or series (model
        parameterization NNs frozen).

        Parameters
        ----------
        dataset_dict
            dictionary containing input data.
        eval
            Whether to run the model in evaluation mode with gradients
            disabled.

        Returns
        -------
        dict[str, torch.Tensor]
            dictionary of model outputs. Each key corresponds to a model name.
        """
        # MoE forward: frozen experts + trainable gate.
        if self.multimodel_type == 'moe':
            return self._forward_moe(dataset_dict, eval)

        self.output_dict = {}
        for name, model in self.model_dict.items():
            if eval:
                ## Inference mode
                model.eval()
                with torch.no_grad():
                    self.output_dict[name] = model(dataset_dict)
            else:
                ## Training mode
                model.train()
                self.output_dict[name] = model(dataset_dict)

        return self.output_dict

    def _forward_moe(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for MoE: run frozen experts + trainable gate.

        Parameters
        ----------
        dataset_dict
            dictionary containing input data.
        eval
            Whether to run in evaluation mode.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary keyed by ``'MoE'`` containing combined predictions.
        """
        if eval:
            self.moe.eval()
            with torch.no_grad():
                combined = self.moe(dataset_dict)
        else:
            self.moe.train()  # Gate trains; experts stay frozen.
            combined = self.moe(dataset_dict)

        self.output_dict = {'MoE': combined}
        return self.output_dict

    def resolve_target_key(self, output_keys) -> str:
        """Resolve the configured target name to a model output key.

        Datasets may name the target differently (e.g. 'QObs') from the
        physics model's standard output key (e.g. 'streamflow'), so the
        alias map in `_target_output_aliases` is consulted as a fallback.

        Parameters
        ----------
        output_keys
            Keys available in the model's output dictionary.

        Returns
        -------
        str
            The key to use to index the model output dictionary.
        """
        target_key = self.target_names[0]
        if target_key not in output_keys:
            target_key = self._target_output_aliases.get(target_key, target_key)
        if target_key not in output_keys:
            raise ValueError(
                f"Target variable '{self.target_names[0]}' not in model outputs "
                f"{list(output_keys)}.",
            )
        return target_key

    def calc_loss(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Calculate combined loss across all models.

        Parameters
        ----------
        dataset_dict
            dictionary containing input data.
        loss_func
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Combined loss across all models.

        TODO: Support different loss functions for each model in ensemble.
        """
        if not self.loss_func and not loss_func:
            raise ValueError("No loss function defined.")
        loss_func = loss_func or self.loss_func

        # MoE: loss on the gated combined output only.
        if self.multimodel_type == 'moe':
            return self._calc_loss_moe(dataset_dict, loss_func)

        loss_combined = 0.0

        # Loss calculation for each model
        for name, output in self.output_dict.items():
            target_key = self.resolve_target_key(output.keys())
            output = output[target_key]

            loss = loss_func(
                output.squeeze(),
                dataset_dict['target'].squeeze(),
                sample_ids=dataset_dict['batch_sample'],
            )
            loss_combined += loss
            self.loss_dict[name] += loss.item()

        return loss_combined

    def _calc_loss_moe(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Calculate loss for MoE on the gated combined output.

        Parameters
        ----------
        dataset_dict
            dictionary containing input data.
        loss_func
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Loss on the gated ensemble predictions.
        """
        output = self.output_dict['MoE']
        target_key = self.resolve_target_key(output.keys())

        pred = output[target_key]
        main_loss = loss_func(
            pred.squeeze(),
            dataset_dict['target'].squeeze(),
            sample_ids=dataset_dict['batch_sample'],
        )

        # Auxiliary losses for MoE.
        if not self.moe.training:
            self.loss_dict['MoE'] += main_loss.item()
            return main_loss

        # MCL mode: delegate to MoE's MCL loss computation.
        if self.moe._use_mcl:
            loss = self.moe.compute_mcl_loss(
                dataset_dict['target'],
                shared_loss_fn=loss_func,
                batch_sample=dataset_dict['batch_sample'],
            )
            self.loss_dict['MoE'] += loss.item()
            return loss

        # Collect all active loss terms.
        # When AWL is enabled, these are combined via learned weights.
        # Otherwise, they are combined with manual weights as before.
        use_awl = self.moe._use_awl and self.moe.awl is not None

        if use_awl:
            # AWL path: collect individual losses (unweighted).
            awl_losses: list[torch.Tensor] = []

            # Main loss (phased weight still applies as a gate).
            main_w = getattr(self.moe, '_main_loss_weight', 1.0)
            awl_losses.append(main_w * main_loss)

            # Oracle loss.
            oracle_weight = getattr(
                self.moe,
                '_oracle_loss_weight_eff',
                self.moe.oracle_loss_weight,
            )
            if oracle_weight > 0:
                awl_losses.append(
                    self.moe.compute_oracle_loss(dataset_dict['target']),
                )

            # Expert specialization losses.
            if self.moe._use_specialization:
                if self.moe.expert_loss_weight > 0:
                    awl_losses.append(
                        self.moe.compute_expert_specialization_loss(
                            dataset_dict['target'],
                        ),
                    )
                if self.moe.diversity_weight > 0:
                    awl_losses.append(
                        self.moe.compute_diversity_loss(dataset_dict['target']),
                    )
                if self.moe.load_balance_weight > 0:
                    awl_losses.append(self.moe.compute_load_balance_loss())
                if self.moe.quality_floor_weight > 0:
                    awl_losses.append(
                        self.moe.compute_expert_quality_loss(
                            dataset_dict['target'],
                        ),
                    )

            # Entropy / error-pred.
            entropy_weight = getattr(
                self.moe,
                '_entropy_reg_weight_eff',
                self.moe.entropy_reg_weight,
            )
            if entropy_weight > 0:
                awl_losses.append(self.moe.compute_entropy_reg())
            if getattr(self.moe, 'error_pred_weight', 0.0) > 0:
                awl_losses.append(
                    self.moe.compute_error_pred_loss(dataset_dict['target']),
                )

            loss = self.moe.awl(*awl_losses)
        else:
            # Manual-weight path (original behaviour).
            main_w = getattr(self.moe, '_main_loss_weight', 1.0)
            loss = main_w * main_loss

            oracle_weight = getattr(
                self.moe,
                '_oracle_loss_weight_eff',
                self.moe.oracle_loss_weight,
            )
            if oracle_weight > 0:
                oracle_loss = self.moe.compute_oracle_loss(
                    dataset_dict['target'],
                )
                loss = loss + oracle_weight * oracle_loss

            entropy_weight = getattr(
                self.moe,
                '_entropy_reg_weight_eff',
                self.moe.entropy_reg_weight,
            )
            if entropy_weight > 0:
                loss = loss + entropy_weight * self.moe.compute_entropy_reg()

            error_pred_weight = getattr(self.moe, 'error_pred_weight', 0.0)
            if error_pred_weight > 0:
                loss = loss + error_pred_weight * self.moe.compute_error_pred_loss(
                    dataset_dict['target'],
                )

            uniform_weight = self.moe.uniform_reg_weight
            if uniform_weight > 0:
                loss = loss + uniform_weight * self.moe.compute_uniform_reg()

            if self.moe._use_specialization:
                if self.moe.expert_loss_weight > 0:
                    loss = (
                        loss
                        + self.moe.expert_loss_weight
                        * self.moe.compute_expert_specialization_loss(
                            dataset_dict['target'],
                        )
                    )
                if self.moe.diversity_weight > 0:
                    loss = (
                        loss
                        + self.moe.diversity_weight
                        * self.moe.compute_diversity_loss(dataset_dict['target'])
                    )
                if self.moe.load_balance_weight > 0:
                    loss = (
                        loss
                        + self.moe.load_balance_weight
                        * self.moe.compute_load_balance_loss()
                    )
                if self.moe.quality_floor_weight > 0:
                    loss = (
                        loss
                        + self.moe.quality_floor_weight
                        * self.moe.compute_expert_quality_loss(
                            dataset_dict['target'],
                        )
                    )

        self.loss_dict['MoE'] += loss.item()
        return loss

    def save_model(self, epoch: int) -> None:
        """Save model state dicts.

        Parameters
        ----------
        epoch
            Epoch number to save model at.
        """
        if self.multimodel_type == 'moe':
            # Save gate weights.
            save_model(self.config['model_dir'], self.moe.gate, 'moe_gate', epoch)
            # Save unfrozen expert weights.
            for name in self.moe._unfrozen_expert_names:
                save_model(
                    self.config['model_dir'],
                    self.moe.experts[name],
                    f'expert_{name}',
                    epoch,
                )
            # Save AWL state if active.
            if self.moe.awl is not None:
                save_model(
                    self.config['model_dir'],
                    self.moe.awl,
                    'moe_awl',
                    epoch,
                )
            # Save metadata for reproducibility.
            meta = {
                'expert_specs': self.config['moe']['experts'],
                'gate_config': self.config['moe'].get('gate', {}),
                'scaling_function': self.config['moe'].get(
                    'scaling_function', 'softmax'
                ),
            }
            torch.save(
                meta,
                os.path.join(self.config['model_dir'], f'moe_meta_ep{epoch}.pt'),
            )
            return

        for name, model in self.model_dict.items():
            save_model(self.config['model_dir'], model, name, epoch)

        if self.verbose:
            log.info(f"All states saved for ep:{epoch}")

    def get_states(self) -> None:
        """
        Helper function to expose physical and hidden (non-trainable) nn model
        states (e.g., for sequential simulations).
        """
        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]
            nn_states = self.model_dict[name].nn_model.get_states()
            try:
                phy_states = self.model_dict[name].phy_model.get_states()
            except AttributeError:
                phy_states = None

            return nn_states, phy_states
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )

    def load_states(
        self,
        *,
        path: Optional[str] = None,
        nn_states: Optional[tuple[torch.Tensor, ...]] = None,
        phy_states: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> None:
        """
        Helper function to load physical and hidden (non-trainable) nn model
        states (e.g., for sequential simulations).
        """
        if path:
            if path and nn_states and phy_states:
                raise ValueError(
                    "Provide either `path` or `nn_states` and `phy_states`, not both.",
                )
            if not os.path.exists(path):
                raise FileNotFoundError(f"State path {path} not found.")

            state_dict = torch.load(path, map_location=self.device)
            nn_states = state_dict.get('nn_states', None)
            phy_states = state_dict.get('phy_states', None)
            if self.verbose:
                log.info(
                    f"Loaded states from file | "
                    f"epoch: {state_dict.get('epoch', 'N/A')} | "
                    f"Resume from timestep: {state_dict.get('last_timestep', 'N/A')}",
                )
        elif nn_states:
            if not isinstance(nn_states, tuple):
                raise ValueError("`nn_states` must be a tuple of tensors.")
        elif phy_states:
            if not isinstance(phy_states, tuple):
                raise ValueError("`phy_states` must be a tuple of tensors.")
        else:
            raise ValueError(
                "Either `path` or `nn_states` and `phy_states` must be provided.",
            )

        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]
            self.model_dict[name].nn_model.load_states(nn_states)

            if phy_states is not None:
                try:
                    self.model_dict[name].phy_model.load_states(phy_states)
                except AttributeError:
                    pass
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )

    def save_states(self) -> None:
        """
        Helper function to save physical and nn model states (trainable and
        non-trainable) to disk.
        """
        if 'test' in self.config['mode']:
            mode = 'test'
        else:
            mode = 'sim'
        time = self.config[mode]['end_time']

        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]

            nn_states, phy_states = self.get_states()

            state_dict = {
                'nn_states': nn_states,
                'nn_trainable': self.model_dict[
                    name
                ].state_dict(),  # weights and biases
                'phy_states': phy_states,
                'epoch': self.epoch,
                'last_timestep': time if time else 'N/A',
            }
            torch.save(state_dict, self.config['model_dir'] + "model_states.pt")
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )
        torch.save(state_dict, self.config['model_dir'] + "model_states.pt")
