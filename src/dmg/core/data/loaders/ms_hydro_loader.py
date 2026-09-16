import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import zarr
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

from dmg.core.data.loaders.base import BaseLoader

log = logging.getLogger(__name__)


class MsHydroLoader(BaseLoader):
    """Data loader for multiscale hydrological data loading.

    All data is read from Zarr store and loaded as PyTorch tensors. According to
    config settings, generate:
    - `dataset` for model inference,
    - `train_dataset` for training,
    - `eval_dataset` for testing.

    Parameters
    ----------
    config
        Configuration dictionary.
    test_split
        Whether to split data into training and testing sets.
    overwrite
        Whether to overwrite existing normalization statistics.
    """

    def __init__(
        self,
        config: dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.supported_data = [
            'merit',
            'merit_71',
            'hf22',
            'hf22_500',
        ]  # Add new supported observation names here.
        self.data_name = config['observations']['name']
        self.nn_attributes = config['model']['nn'].get('attributes', [])
        self.nn_forcings = config['model']['nn'].get('forcings', [])
        self.phy_attributes = config['model']['phy'].get('attributes', [])
        self.phy_forcings = config['model']['phy'].get('forcings', [])
        self.all_forcings = self.config['observations']['all_forcings']
        self.all_attributes = self.config['observations']['all_attributes']

        self.target = config['train']['target']
        self.log_norm_vars = config['model'].get('use_log_norm', [])
        self.attr_group_name = self.config['observations'].get(
            'attr_group_name',
            'attrs',
        )
        self.device = config['device']
        self.dtype = config['dtype']

        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None

        if self.data_name not in self.supported_data:
            raise ValueError(f"Data source '{self.data_name}' not supported.")

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset into dictionary of nn and physics model input arrays."""
        mode = self.config['mode']
        if mode == 'sim':
            self.dataset = self._preprocess_data(scope='simulation')
        elif self.test_split:
            self.train_dataset = self._preprocess_data(scope='train')
            self.eval_dataset = self._preprocess_data(scope='test')
        elif mode in ['train', 'test']:
            self.train_dataset = self._preprocess_data(scope=mode)
        else:
            self.dataset = self._preprocess_data(scope='all')

    def _preprocess_data(
        self,
        scope: Optional[str],
    ) -> dict[str, torch.Tensor]:
        """Read data, preprocess, and return as tensors for models."""
        # Load target data and gage-cat mapping when available.
        obs_config = self.config['observations']
        has_target = 'target_path' in obs_config and 'cat_idx_path' in obs_config

        if has_target:
            target, gage_key, cat_idx = self._load_target_data(scope)
        else:
            target = None
            gage_key = None
            cat_idx = {}

        # Subsetting gages: dynamically load only necessary cats instead of full
        # zarr. Massive io saver.
        cat_subset = None
        is_subset = (
            obs_config.get('subset_path') is not None
            or self.config['train'].get('n_test_gages') is not None
        )
        if is_subset and cat_idx:
            needed = set()
            for idxs in cat_idx.values():
                needed.update(idxs)
            cat_subset = np.sort(np.array(list(needed)))
            log.info(
                f"Subsetting to {len(cat_subset)} catchments for {len(cat_idx)} gages",
            )

        ac_all, elev_all, ai_all, subbasin_id_all, x_nn, c_nn = self.read_data(
            scope, cat_indices=cat_subset
        )

        # Re-map cat_idx to contiguous indices matching the subsetted arrays.
        if cat_subset is not None and cat_idx:
            remap = np.empty(int(cat_subset.max()) + 1, dtype=np.intp)
            remap[cat_subset] = np.arange(len(cat_subset))
            cat_idx = {
                gk: remap[np.array(idxs, dtype=np.intp)].tolist()
                for gk, idxs in cat_idx.items()
            }

        # Validate gage-catchment alignment between target and forcing data.
        if has_target and cat_idx and gage_key is not None:
            target, gage_key, cat_idx = self._validate_gage_cat_alignment(
                target,
                gage_key,
                cat_idx,
                n_catchments=x_nn.shape[0],
            )

        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn)
        x_nn_norm, c_nn_norm = self.normalize(x_nn, c_nn)

        np.nan_to_num(x_nn, copy=False)

        dataset = {
            'ac_all': ac_all,
            'elev_all': elev_all,
            'ai_all': ai_all,
            'subbasin_id_all': subbasin_id_all,
            'x_nn_norm': x_nn_norm,
            'c_nn_norm': c_nn_norm,
            'x_phy': np.swapaxes(x_nn, 1, 0),
            'target': target,
        }

        # Include gage-cat mapping for any scope when available.
        # Sim-only configs (e.g. merit71) may not have target/cat_idx.
        if gage_key is not None:
            dataset['gage_key'] = gage_key
        if cat_idx:
            dataset['cat_idx'] = cat_idx

        return dataset

    def read_data(
        self,
        scope: Optional[str],
        cat_indices: Optional[NDArray] = None,
    ) -> tuple[NDArray[np.float32]]:
        """Read data from the data file.

        Parameters
        ----------
        scope
            Scope of data to read, affects what timespan of data is loaded.
        cat_indices
            Optional sorted array of catchments indices to load. When
            provided, only these catchments are read from zarr store to save on
            io and memory.

        Returns
        -------
        tuple[NDArray[np.float32]]
            tuple of neural network + physics model inputes, and target data.
        """
        try:
            if scope == 'train':
                time = self.config['train_time']
            elif scope == 'test':
                time = self.config['test_time']
            elif scope == 'simulation':
                time = self.config['sim_time']
            elif scope == 'all':
                time = self.config['all_time']
            else:
                raise ValueError(
                    "Scope must be 'train', 'test', 'simulation', or 'all'.",
                )
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.") from e

        # Get time indicies
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1],
            freq='d',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        root_zone = zarr.open_group(
            self.config['observations']['data_path'],
            mode='r',
        )
        raw_ids = root_zone[self.config['observations']['subbasin_id_name']][:]
        subbasin_id_all = np.array(raw_ids)

        # Apply basin subsetting
        if cat_indices is not None:
            subbasin_id_all = subbasin_id_all[cat_indices]
            n_basins = len(cat_indices)
        else:
            n_basins = root_zone[self.nn_forcings[0]].shape[0]

        # Forcing subset
        n_time = idx_end - idx_start
        forc_array = np.empty(
            (n_basins, n_time, len(self.nn_forcings)),
            dtype=np.float32,
        )
        for i, forc in enumerate(self.nn_forcings):
            if forc not in self.all_forcings:
                raise ValueError(f"Forcing {forc} not listed in available forcings.")
            if cat_indices is not None:
                forc_array[:, :, i] = root_zone[forc].oindex[
                    cat_indices,
                    idx_start:idx_end,
                ]
            else:
                forc_array[:, :, i] = root_zone[forc][:, idx_start:idx_end]

        # Attribute subset
        attr_group = root_zone[self.attr_group_name]
        _sel = cat_indices if cat_indices is not None else slice(None)
        attr_array = np.empty(
            (n_basins, len(self.nn_attributes)),
            dtype=np.float32,
        )
        for i, attr in enumerate(self.nn_attributes):
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            attr_array[:, i] = attr_group[attr][:][_sel]

        # Get static attributes (upstream area, elevation, subbasin area)
        try:
            ac_name = self.config['observations']['upstream_area_name']
            ac_array = attr_group[ac_name][:].astype(np.float32)[_sel]
        except (ValueError, KeyError) as e:
            raise ValueError("Upstream area is not provided.") from e

        try:
            elevation_name = self.config['observations']['elevation_name']
            elev_array = attr_group[elevation_name][:].astype(np.float32)[_sel]
        except (ValueError, KeyError) as e:
            raise ValueError("Elevation is not provided.") from e

        try:
            ai_name = self.config['observations']['subbasin_area_name']
            ai_array = attr_group[ai_name][:].astype(np.float32)[_sel]
        except (ValueError, KeyError) as e:
            raise ValueError("Subbasin area (catchment size) is not provided.") from e

        return [
            ac_array,
            elev_array,
            ai_array,
            subbasin_id_all,
            forc_array,
            attr_array,
        ]

    def _load_target_data(
        self,
        scope: str,
    ) -> tuple:
        """Load streamflow observations and gage-catchment mapping.

        Parameters
        ----------
        scope
            Data scope (e.g. 'train') -- determines time window.

        Returns
        -------
        tuple
            (target, gage_key, cat_idx) where `target` has shape
            (n_time, n_gages, 1) in mm/day, `gage_key` is a list of
            gage ID strings, and `cat_idx` maps each gage ID to a list
            of integer indices into the catchment arrays.
        """
        obs_config = self.config['observations']

        # Gage-cat map
        with open(obs_config['cat_idx_path']) as f:
            cat_idx_raw = json.load(f)

        gage_list = [k.zfill(8) for k in cat_idx_raw]
        gage_list.sort()

        # Optional gage subsetting via explicit list or random selection.
        subset_path = obs_config.get('subset_path', None)
        n_test_gages = self.config['train'].get('n_test_gages', None)

        if subset_path is not None:
            # Subset by file
            with open(subset_path) as f:
                subset_gages = json.load(f)
            subset_gages = [str(g).zfill(8) for g in subset_gages]
            gage_list = [g for g in gage_list if g in subset_gages]
            if not gage_list:
                raise ValueError(
                    f"No gages from subset_path '{subset_path}' found in "
                    f"cat_idx_path. Check gage ID formatting.",
                )
            log.info(f"Subsetting to {len(gage_list)} gages from {subset_path}")
        elif n_test_gages is not None and n_test_gages < len(gage_list):
            # Subset by random selection -- for debug
            log.info(
                f"Randomly subsetting from {len(gage_list)} gages to {n_test_gages} gages for debug."
            )
            rng = np.random.default_rng(seed=42)
            gage_list = sorted(
                rng.choice(
                    gage_list,
                    size=n_test_gages,
                    replace=False,
                ).tolist()
            )
            log.info(f"Randomly subsetting to {n_test_gages} gages")

        # Streamflow obs
        obs_root = zarr.open_group(obs_config['target_path'], mode='r')
        obs_gage_ids = obs_root['GAGEID'][:]
        observation = obs_root['observation'][:]

        # Intersect gage list with available observations
        _, idx_gage_list, idx_obs = np.intersect1d(
            gage_list,
            obs_gage_ids,
            return_indices=True,
        )

        # Slice obs to time window
        obs_all_time = pd.date_range(
            obs_config['target_start_time'],
            obs_config['target_end_time'],
            freq='d',
        )
        if scope == 'train':
            time = self.config['train_time']
        elif scope == 'test':
            time = self.config['test_time']
        elif scope == 'simulation':
            time = self.config['sim_time']
        else:
            time = self.config['all_time']

        idx_start = obs_all_time.get_loc(time[0])
        idx_end = obs_all_time.get_loc(time[-1]) + 1

        streamflow = observation[idx_obs, idx_start:idx_end]

        # Basin drainage area for unit conversion (ft3 s-1 -> mm d-1)
        basin_area = np.array(
            obs_root.attrs['Drainage area (km^2)'],
        )[idx_obs]

        area_tiled = np.tile(
            basin_area[:, np.newaxis],
            (1, streamflow.shape[1]),
        )
        flow_mm = streamflow * 0.0283168 * 3600 * 24 * 1e3 / (area_tiled * 1e6)

        target = np.swapaxes(flow_mm[:, :, np.newaxis], 0, 1)  # [nt, nb, 1]

        # Matched gage-cat index mapping
        gage_key = list(np.array(gage_list)[idx_gage_list])
        cat_idx = {}
        for gk in gage_key:
            raw_key = gk.lstrip('0') or '0'
            if gk in cat_idx_raw:
                cat_idx[gk] = [int(v) for v in cat_idx_raw[gk]]
            elif raw_key in cat_idx_raw:
                cat_idx[gk] = [int(v) for v in cat_idx_raw[raw_key]]

        return target, gage_key, cat_idx

    def _validate_gage_cat_alignment(
        self,
        target: NDArray[np.float32],
        gage_key: list[str],
        cat_idx: dict[str, list[int]],
        n_catchments: int,
    ) -> tuple[NDArray[np.float32], list[str], dict[str, list[int]]]:
        """Validate and fix alignment between target gages and catchment data.

        Checks that every gage in ``gage_key`` has a ``cat_idx`` entry whose
        indices are within bounds of the loaded catchment arrays. Gages that
        fail validation are dropped from ``target``, ``gage_key``, and
        ``cat_idx``.

        Parameters
        ----------
        target
            Observation array, shape ``[n_time, n_gages, 1]``.
        gage_key
            List of gage ID strings aligned with the gage axis of *target*.
        cat_idx
            Mapping of gage ID to list of catchment indices.
        n_catchments
            Number of catchments in the loaded forcing/attribute arrays.

        Returns
        -------
        tuple
            Validated (target, gage_key, cat_idx).
        """
        valid_mask = []
        invalid_gages: list[str] = []

        for gk in gage_key:
            if gk not in cat_idx:
                valid_mask.append(False)
                invalid_gages.append(gk)
                continue

            idxs = cat_idx[gk]
            if any(i < 0 or i >= n_catchments for i in idxs):
                valid_mask.append(False)
                invalid_gages.append(gk)
                continue

            valid_mask.append(True)

        if invalid_gages:
            log.warning(
                f"Dropping {len(invalid_gages)} gages with missing or "
                f"out-of-bounds cat_idx entries: {invalid_gages[:10]}"
                + (
                    f" ... ({len(invalid_gages)} total)"
                    if len(invalid_gages) > 10
                    else ""
                ),
            )
            keep = np.array(valid_mask)
            target = target[:, keep, :]
            gage_key = [gk for gk, v in zip(gage_key, valid_mask) if v]
            cat_idx = {gk: cat_idx[gk] for gk in gage_key}

        if len(gage_key) == 0:
            raise ValueError(
                "No valid gage-catchment mappings remain after validation. "
                "Check that cat_idx_path is consistent with the forcing zarr.",
            )

        return target, gage_key, cat_idx

    def load_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> None:
        """Load or calculate normalization statistics if necessary.

        Parameters
        ----------
        x_nn
            Neural network dynamic data [n_basins, n_time, n_forcing].
        c_nn
            Neural network static data [n_basins, n_attr].
        """
        self.out_path = os.path.join(
            self.config['model_dir'],
            'normalization_statistics.json',
        )

        if os.path.isfile(self.out_path) and (not self.overwrite):
            if not self.norm_stats:
                with open(self.out_path) as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(x_nn, c_nn)

    def _init_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> dict[str, list[float]]:
        """Compile and save normalization statistics for forcings and attributes."""
        stat_dict = {}

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)

        return stat_dict

    def _calc_norm_stats(self, x: NDArray[np.float32]) -> list[float]:
        """Calculate Gaussian normalization statistics.

        Flatten, exclude NaNs, then return [p10, p90, mean, std].

        Parameters
        ----------
        x
            Input data array.

        Returns
        -------
        list[float]
            [10th percentile, 90th percentile, mean, std].
        """
        a = x.flatten()
        b = a[~np.isnan(a)]
        if b.size == 0:
            b = np.array([0])

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x: NDArray[np.float32]) -> list[float]:
        """Calculate log-sqrt (gamma) normalization statistics.

        Flatten, exclude NaNs, apply log10(sqrt(x) + 0.1) transform, then return
        [p10, p90, mean, std].

        Parameters
        ----------
        x
            Input data array.

        Returns
        -------
        list[float]
            [10th percentile, 90th percentile, mean, std].
        """
        a = x.flatten()
        b = a[~np.isnan(a)]
        b = np.log10(np.sqrt(b) + 0.1)

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Normalize data for neural network.

        Parameters
        ----------
        x_nn
            Neural network dynamic data.
        c_nn
            Neural network static data.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            Normalized forcings x_nn_norm and attributes c_nn_norm,
            stored separately to avoid tiling attributes over the full time
            dimension.
        """
        x_nn_norm = self._to_norm(x_nn, self.nn_forcings)
        c_nn_norm = self._to_norm(c_nn, self.nn_attributes)

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        del x_nn

        return x_nn_norm, c_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Normalize data with Gaussian or log-Gaussian norm."""
        data_norm = np.zeros(data.shape, dtype=data.dtype)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]

            if len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        # NOTE: Should be external, except altering order of first two dims
        # augments normalization...
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

    def _from_norm(
        self,
        data_scaled: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """De-normalize data with a Gaussian or log-Gaussian norm."""
        data = np.zeros(data_scaled.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if len(data_scaled.shape) == 3:
                data[:, :, k] = data_scaled[:, :, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
            elif len(data_scaled.shape) == 2:
                data[:, k] = data_scaled[:, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)

    def _fill_nan(self, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fill NaN values in a 3D array with linear interpolation."""
        # Define the x-axis for interpolation
        x = np.arange(array.shape[1])

        # Iterate over the 1st and 3rd dims to interpolate the 2nd dim
        for i in range(array.shape[0]):
            for j in range(array.shape[2]):
                slice_1d = array[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Conditional linear interpolation
                if np.any(nans) and (np.sum(non_nans) > 1):
                    array[i, :, j] = np.interp(
                        x,
                        x[non_nans],
                        slice_1d[non_nans],
                        left=None,
                        right=None,
                    )
        return array
