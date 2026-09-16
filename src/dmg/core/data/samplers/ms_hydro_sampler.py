import logging
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler

log = logging.getLogger(__name__)


class MsHydroSampler(BaseSampler):
    """Multiscale hydrological data sampler.

    Constructs training batches by sampling gages and gathering their
    constituent catchement data for multiscale model training.

    Parameters
    ----------
    config
        Configuration dictionary.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warmup = config['model']['warmup']
        self.rho = config['model']['rho']

    def select_subset(
        self,
        x: NDArray[np.float32],
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
        warmup: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Select a subset of input array for gage-level data.

        Handles temporal subsetting with random time indices per sample.

        Parameters
        ----------
        x
            Input data array [nt, nb, nvar] or [nb, nvar].
        i_grid
            Gage indices to select.
        i_t
            Time start indices for each sample.
        c
            Optional static data to concatenate.
        tuple_out
            If True, return a tuple of (x_tensor, c_tensor).
        has_grad
            If True, create tensors with gradient tracking.
        warmup
            Override for the warm-up window length. When None, uses
            self.warmup. Pass 0 for target data which should not
            include the warm-up period (matching the reference
            selectSubset(y, iGrid, iT, rho) call).
        device
            Device to place output tensors on.
        """
        device = device if device is not None else self.device
        warmup = warmup if warmup is not None else self.warmup
        batch_size = len(i_grid)
        nx = x.shape[-1]

        if i_t is not None:
            x_tensor = torch.zeros(
                [self.rho + warmup, batch_size, nx],
                device=device,
                requires_grad=has_grad,
            )
            for k in range(batch_size):
                x_tensor[:, k : k + 1, :] = torch.as_tensor(
                    x[
                        i_t[k] - warmup : i_t[k] + self.rho,
                        i_grid[k] : i_grid[k] + 1,
                        :,
                    ],
                    dtype=torch.float32,
                )
        else:
            if x.ndim == 3:
                x_tensor = torch.as_tensor(x[:, i_grid, :], dtype=torch.float32)
            else:
                x_tensor = torch.as_tensor(x[i_grid, :], dtype=torch.float32)

        if c is not None:
            c_tensor = torch.as_tensor(
                c[i_grid],
                dtype=torch.float32,
            )
            c_tensor = c_tensor.unsqueeze(1).repeat(1, self.rho + warmup, 1)
            if tuple_out:
                return (
                    x_tensor.to(device),
                    c_tensor.to(device),
                )
            return torch.cat(
                (x_tensor, c_tensor),
                dim=2,
            ).to(device)

        return x_tensor.to(device)

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
        device: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Generate a multiscale training batch.

        Randomly samples gages and time windows, then gathers catchment data for
        each selected gage. If the total number of catchments exceeds
        max_cat_size, the batch is truncated.

        Expected dataset keys
        ---------------------
        x_nn_norm : [nt, n_cat, n_forc]
            Normalized cat-level forcings for the NN temporal branch.
        c_nn_norm : [n_cat, n_attr]
            Normalized cat-level static attributes.
        target : [nt, n_gage, ny]
            Gage-level observation target.
        gage_key : list[str]
            List of gage identifiers.
        cat_idx : dict[str, list[int]]
            Mapping from gage key to catchment indices.
        Ac_all : [n_cat,]
            Upstream area for each catchment.
        Ai_all : [n_cat,]
            Unit area for each catchment.
        Ele_all : [n_cat,]
            Mean elevation for each catchment.

        Parameters
        ----------
        dataset
            Training dataset dictionary.
        ngrid_train
            Number of gages in the training dataset.
        nt
            Number of timesteps in the training dataset.
        device
            Device to place output tensors on. Batch prefetch uses CPU to avoid
            GPU vram competition.

        Returns
        -------
        dict[str, torch.Tensor]
            Training batch dictionary with keys:
            x_phy, xc_nn_norm, c_nn_norm, target,
            ac_all, elev_all, areas, outlet_topo,
            batch_sample.
        """
        batch_size = self.config['train']['batch_size']
        max_cat_size = self.config['train'].get('max_cat_size', 1000)

        # Random sample gage and time indices.
        i_grid, i_t = random_index(
            ngrid_train,
            nt,
            (batch_size, self.rho),
            warmup=self.warmup,
        )

        gage_key = dataset['gage_key']
        cat_idx = dataset['cat_idx']
        gage_key_batch = np.array(gage_key)[i_grid]

        # Build catchment index lists, clip to max batch size to prevent vram overallocation.
        id_list = []
        start_id = 0
        for gage_idx, gage in enumerate(gage_key_batch):
            n_cat_gage = len(cat_idx[gage])
            if (start_id + n_cat_gage) > max_cat_size:
                i_grid = i_grid[:gage_idx]
                i_t = i_t[:gage_idx]
                gage_key_batch = gage_key_batch[:gage_idx]
                break
            id_list.append(range(start_id, start_id + n_cat_gage))
            start_id += n_cat_gage

        n_cat_total = start_id
        actual_batch = len(gage_key_batch)

        # Pad catchment dim to max_cat_size for fixed-shape batches.
        pad_batch = self.config['train'].get('pad_batch', False)
        alloc_size = max_cat_size if pad_batch else n_cat_total
        if pad_batch:
            log.debug(
                f"pad_batch: {n_cat_total}/{max_cat_size} cats used "
                f"({n_cat_total / max_cat_size:.1%} fill)"
            )

        # Select gage-level target: [rho, nb, ny].
        # NOTE: warmup=0 removes warmup from target.
        target = self.select_subset(
            dataset['target'],
            i_grid,
            i_t,
            warmup=0,
            device=device,
        )

        forcing_norm = np.asarray(dataset['x_nn_norm'])
        forcing_raw = np.asarray(dataset['x_phy'])
        attr_norm = np.asarray(dataset['c_nn_norm'])
        Ac_all = np.asarray(dataset['ac_all'])
        Ai_all = np.asarray(dataset['ai_all'])
        Ele_all = np.asarray(dataset['elev_all'])

        n_forcing = forcing_norm.shape[-1]
        n_attr = attr_norm.shape[-1]
        rho_wu = self.rho + self.warmup

        # Batch arrays (alloc_size == max_cat_size when pad_batch is True).
        # When padding, init with zeros so the padded tail is already correct;
        # NaN cleanup only runs on the real-data slice [0:n_cat_total].
        fill_val = 0.0 if pad_batch else np.nan
        xTrain2 = np.full(
            (alloc_size, rho_wu, n_forcing),
            fill_val,
        )
        xTrain2_raw = np.full(
            (alloc_size, rho_wu, n_forcing),
            fill_val,
        )
        attr2 = np.full((alloc_size, n_attr), fill_val)
        idx_matrix = np.zeros((alloc_size, actual_batch))
        Ai_batch = []
        Ac_batch = []
        Ele_batch = []

        for gageidx, gage in enumerate(gage_key_batch):
            merit_indices = np.array(cat_idx[gage]).astype(int)
            id_range = np.array(id_list[gageidx])

            idx_matrix[id_range, gageidx] = 1

            # Normalize unit areas within each gage's catchments
            ai_vals = Ai_all[merit_indices]
            Ai_batch.extend(ai_vals / ai_vals.sum())
            Ac_batch.extend(Ac_all[merit_indices])
            Ele_batch.extend(Ele_all[merit_indices])

            t_start = i_t[gageidx] - self.warmup
            t_end = i_t[gageidx] + self.rho

            # Forcings
            xTrain2[id_range, :, :] = np.swapaxes(
                forcing_norm[t_start:t_end, merit_indices, :],
                0,
                1,
            )
            xTrain2_raw[id_range, :, :] = np.swapaxes(
                forcing_raw[t_start:t_end, merit_indices, :],
                0,
                1,
            )
            attr2[id_range, :] = attr_norm[merit_indices, :]

        # Pad area/elevation lists to alloc_size (zeros for padded catchments).
        if pad_batch:
            n_pad = alloc_size - n_cat_total
            Ai_batch.extend([0.0] * n_pad)
            Ac_batch.extend([0.0] * n_pad)
            Ele_batch.extend([0.0] * n_pad)

        # Replace NaN --> 0 (only in real-data slice when padding, since
        # the padded tail is already zero-initialized).
        if pad_batch:
            s = slice(None, n_cat_total)
            xTrain2[s][np.isnan(xTrain2[s])] = 0
            xTrain2_raw[s][np.isnan(xTrain2_raw[s])] = 0
            attr2[s][np.isnan(attr2[s])] = 0
        else:
            xTrain2[np.isnan(xTrain2)] = 0
            xTrain2_raw[np.isnan(xTrain2_raw)] = 0
            attr2[np.isnan(attr2)] = 0

        device = device if device is not None else self.device

        x_phy = (
            torch.from_numpy(
                np.swapaxes(xTrain2_raw, 0, 1),
            )
            .float()
            .to(device)
        )

        # Static attrs
        c_nn_norm = torch.from_numpy(attr2).float().to(device)

        # Combined NN input: expand attrs over time and concat with forcings
        xTrain2_torch = torch.from_numpy(xTrain2).float().to(device)
        attr_expand = c_nn_norm.unsqueeze(1).expand(
            -1,
            self.rho + self.warmup,
            -1,
        )
        xc_nn_norm = torch.cat((xTrain2_torch, attr_expand), dim=-1)
        xc_nn_norm = xc_nn_norm.permute(1, 0, 2)

        return {
            'x_phy': x_phy,
            'xc_nn_norm': xc_nn_norm,
            'c_nn_norm': c_nn_norm,
            'target': target,
            'ac_all': torch.tensor(
                Ac_batch,
                dtype=torch.float32,
                device=device,
            ),
            'elev_all': torch.tensor(
                Ele_batch,
                dtype=torch.float32,
                device=device,
            ),
            'areas': torch.tensor(
                Ai_batch,
                dtype=torch.float32,
                device=device,
            ),
            'outlet_topo': torch.from_numpy(idx_matrix).float().to(device),
            'batch_sample': i_grid[:actual_batch],
        }

    def get_eval_sample(
        self,
        dataset: dict[str, torch.Tensor],
        gage_start: int,
        gage_end: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a multiscale evaluation batch grouped by gages.

        Gathers the catchment data for gages in [gage_start, gage_end]
        and builds the area-weighted aggregation matrix needed to convert
        catchment-level model output to gage-level predictions.  Uses the
        full time series (no random time subsetting).

        Parameters
        ----------
        dataset
            Evaluation dataset dictionary.  Must contain gage_key,
            cat_idx, x_nn_norm, c_nn_norm, x_phy,
            ac_all, ai_all, elev_all, and optionally
            target.
        gage_start
            Start index into the gage list (inclusive).
        gage_end
            End index into the gage list (exclusive).

        Returns
        -------
        dict[str, torch.Tensor]
            Evaluation batch with keys x_phy, xc_nn_norm,
            c_nn_norm, target, ac_all, elev_all,
            areas, outlet_topo.
        """
        device = self.config['device']
        gage_key = dataset['gage_key']
        cat_idx = dataset['cat_idx']
        gage_key_batch = gage_key[gage_start:gage_end]
        n_gages_batch = len(gage_key_batch)

        forcing_norm = np.asarray(dataset['x_nn_norm'])
        forcing_raw = np.asarray(dataset['x_phy'])
        attr_norm = np.asarray(dataset['c_nn_norm'])
        Ac_all = np.asarray(dataset['ac_all'])
        Ai_all = np.asarray(dataset['ai_all'])
        Ele_all = np.asarray(dataset['elev_all'])

        n_forcing = forcing_norm.shape[-1]
        n_attr = attr_norm.shape[-1]
        n_time = forcing_norm.shape[0]

        # Build catchment index lists
        id_list = []
        start_id = 0
        for gage in gage_key_batch:
            cat_indices = cat_idx[gage]
            n_cat_gage = len(cat_indices)
            id_list.append(range(start_id, start_id + n_cat_gage))
            start_id += n_cat_gage

        n_cat_total = start_id

        forc_norm_batch = np.full(
            (n_cat_total, n_time, n_forcing),
            np.nan,
        )
        forc_raw_batch = np.full(
            (n_cat_total, n_time, n_forcing),
            np.nan,
        )
        attr_batch = np.full((n_cat_total, n_attr), np.nan)
        idx_matrix = np.zeros((n_cat_total, n_gages_batch))
        Ai_batch = []
        Ac_batch = []
        Ele_batch = []

        for gageidx, gage in enumerate(gage_key_batch):
            cat_indices = np.array(cat_idx[gage]).astype(int)
            id_range = np.array(id_list[gageidx])

            idx_matrix[id_range, gageidx] = 1

            # Normalize unit areas within each gage's catchments
            ai_vals = Ai_all[cat_indices]
            Ai_batch.extend(ai_vals / ai_vals.sum())
            Ac_batch.extend(Ac_all[cat_indices])
            Ele_batch.extend(Ele_all[cat_indices])

            # Forcings
            forc_norm_batch[id_range, :, :] = np.swapaxes(
                forcing_norm[:, cat_indices, :],
                0,
                1,
            )
            forc_raw_batch[id_range, :, :] = np.swapaxes(
                forcing_raw[:, cat_indices, :],
                0,
                1,
            )
            attr_batch[id_range, :] = attr_norm[cat_indices, :]

        # Replace NaN --> 0
        forc_norm_batch[np.isnan(forc_norm_batch)] = 0
        forc_raw_batch[np.isnan(forc_raw_batch)] = 0
        attr_batch[np.isnan(attr_batch)] = 0

        x_phy = (
            torch.from_numpy(
                np.swapaxes(forc_raw_batch, 0, 1),
            )
            .float()
            .to(device)
        )

        # Static attrs
        c_nn_norm = torch.from_numpy(attr_batch).float().to(device)

        # Combined NN input: expand attrs over time and concat with forcings
        forc_torch = torch.from_numpy(forc_norm_batch).float().to(device)
        attr_expand = c_nn_norm.unsqueeze(1).expand(-1, n_time, -1)
        xc_nn_norm = torch.cat((forc_torch, attr_expand), dim=-1)
        xc_nn_norm = xc_nn_norm.permute(1, 0, 2)

        # Target: gage-level, full time series.
        target = None
        if dataset.get('target') is not None:
            target = torch.tensor(
                dataset['target'][:, gage_start:gage_end, :],
                dtype=torch.float32,
                device=device,
            )

        return {
            'x_phy': x_phy,
            'xc_nn_norm': xc_nn_norm,
            'c_nn_norm': c_nn_norm,
            'target': target,
            'ac_all': torch.tensor(
                Ac_batch,
                dtype=torch.float32,
                device=device,
            ),
            'elev_all': torch.tensor(
                Ele_batch,
                dtype=torch.float32,
                device=device,
            ),
            'areas': torch.tensor(
                Ai_batch,
                dtype=torch.float32,
                device=device,
            ),
            'outlet_topo': torch.from_numpy(idx_matrix).float().to(device),
        }

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        dataset_sample = {}
        device = self.config['device']

        for key, value in dataset.items():
            if key in ('x_nn_norm', 'c_nn_norm'):
                continue
            if not hasattr(value, 'dtype') or not np.issubdtype(value.dtype, np.number):
                continue
            if value.ndim == 3:
                if key in ['x_phy']:
                    warmup = 0
                else:
                    warmup = self.config['model']['warmup']
                dataset_sample[key] = torch.tensor(
                    value[warmup:, i_s:i_e, :],
                    dtype=torch.float32,
                    device=device,
                )
            elif value.ndim == 2:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e, :],
                    dtype=torch.float32,
                    device=device,
                )
            elif value.ndim == 1:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Incorrect input dimensions. {key} array must have 1, 2 or 3 dimensions.",
                )

        x_nn_batch = torch.tensor(
            dataset['x_nn_norm'][:, i_s:i_e, :],
            dtype=torch.float32,
            device=device,
        )
        c_nn_batch = torch.tensor(
            dataset['c_nn_norm'][i_s:i_e, :],
            dtype=torch.float32,
            device=device,
        )
        c_nn_expanded = c_nn_batch.unsqueeze(0).expand(
            x_nn_batch.shape[0],
            -1,
            -1,
        )
        dataset_sample['xc_nn_norm'] = torch.cat(
            (x_nn_batch, c_nn_expanded),
            dim=-1,
        )
        dataset_sample['c_nn_norm'] = c_nn_batch

        return dataset_sample
