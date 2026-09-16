"""
Precompute foundation-model embeddings for a dataset and save them to disk,
in the format EmbeddingFinetuneLoader/EmbeddingFinetuneing expect.

This is the offline counterpart to DirectFinetuneing: instead of running the
frozen encoder on-the-fly at every training step, this script runs it once
over a whole pretrain dataset and writes the resulting embeddings to NetCDF.
It reuses the exact same checkpoint-loading and encoding code DirectFinetuneing
uses (dmg.models.neural_networks.direct_finetuneing.build_pretrained_encoder /
encode_with_pretrained), so the embeddings it produces are numerically
identical to what DirectFinetuneing would compute on-the-fly for the same
inputs -- this script just does it once instead of every step.

Because a full daily embedding stream (years x stations x embed_dim floats)
can be very large, this script can additionally aggregate it (mean-pool) to
coarser timelines -- monthly, seasonal, annual -- each written as its own,
much smaller NetCDF file. EmbeddingFinetuneLoader consumes any of these
transparently: it matches each task day to the most recent embedding at or
before it, so a monthly file's embedding simply holds constant across every
day in that month.

Sequence length is bounded with a sliding window (--window_days,
--context_days) rather than encoding the whole record in one shot, since the
encoder's positional table has a fixed patch-token capacity (see
TFTPositionalEncoding's max_len) that a multi-decade daily record can exceed.

Usage
-----
python scripts/generate_embeddings.py \\
    --config conf/templates/_encoder_arch_template.yaml \\
    --out_dir /path/to/embeddings \\
    --dataset_name MyDataset \\
    --resolutions monthly seasonal annual

Reads model.nn.pretrained_time_series_vars/pretrained_static_vars/hidden_size/
num_heads/num_enc_layers/d_ffd/dropout/pretrained_model from --config (the
same fields an existing DirectFinetuneing config already sets), and encodes
config['pretrained_path'] unless --pretrain_data overrides it. Output goes to
<out_dir>/<dataset_name>/<dataset_name>_embeddings_<resolution>.nc
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import netCDF4
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf

# Make both `dmg.*` (package-style) and the bare `models.*`/`core.*` imports
# used internally by the transformer/adapter modules resolvable, regardless
# of whether dmg is pip-installed. Mirrors what happens automatically when
# running `python src/dmg/__main__.py` directly (its own directory lands on
# sys.path[0]).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC = os.path.join(_REPO_ROOT, 'src')
_SRC_DMG = os.path.join(_SRC, 'dmg')
for _p in (_SRC, _SRC_DMG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dmg.core.data.data import extract_temporal_features  # noqa: E402
from dmg.core.data.loader_utils import load_norm_stats, normalize_data  # noqa: E402
from dmg.core.data.loaders.load_nc import NetCDFDataset  # noqa: E402
from dmg.models.neural_networks.direct_finetuneing import (  # noqa: E402
    build_pretrained_encoder,
    encode_with_pretrained,
)

RESOLUTION_CHOICES = ['daily', 'monthly', 'seasonal', 'annual']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', required=True,
        help="dMG yaml config whose model.nn section defines the encoder "
             "architecture (hidden_size, num_heads, num_enc_layers, d_ffd, "
             "dropout, pretrained_time_series_vars, pretrained_static_vars) "
             "and the checkpoint path (pretrained_model).",
    )
    parser.add_argument(
        '--pretrain_data', default=None,
        help="NetCDF file to encode. Defaults to the config's top-level "
             "'pretrained_path'.",
    )
    parser.add_argument(
        '--out_dir', required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        '--dataset_name', required=True,
        help="Subfolder/file stem for this dataset's embedding files.",
    )
    parser.add_argument(
        '--resolutions', nargs='+', default=['monthly', 'seasonal', 'annual'],
        choices=RESOLUTION_CHOICES,
        help="Which timeline aggregations to write. 'daily' is the full-"
             "resolution stream (large); omit it unless you need per-day "
             "embeddings. Default: monthly seasonal annual.",
    )
    parser.add_argument(
        '--embedding_var_name', default='embedding',
        help="Variable name to store the embedding under (must match "
             "model.nn.embedding_var_name at load time).",
    )
    parser.add_argument('--start_time', default=None, help="YYYY-MM-DD, default: file's earliest date")
    parser.add_argument('--end_time', default=None, help="YYYY-MM-DD, default: file's latest date")
    parser.add_argument(
        '--window_days', type=int, default=3650,
        help="Max days encoded in a single forward pass (bounds the "
             "positional-encoding table's patch-token length).",
    )
    parser.add_argument(
        '--context_days', type=int, default=365,
        help="Days of lead-in context prepended to every window after the "
             "first and then discarded, so each window's encoder output "
             "isn't starting cold.",
    )
    parser.add_argument('--batch_size', type=int, default=64, help="Basins per forward pass.")
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--complevel', type=int, default=4, help="NetCDF zlib compression level (0-9).")
    parser.add_argument(
        '--overwrite_norm_stats', action='store_true',
        help="Recompute pretrained-variable normalization stats even if a "
             "cached stats file already exists.",
    )
    return parser.parse_args()


def sliding_windows(
    n_time: int, window_days: int, context_days: int
) -> List[Tuple[int, int, int, int]]:
    """Tile [0, n_time) into (w_start, w_end, keep_start, keep_end) windows.

    Every window after the first is fed `context_days` of lead-in that gets
    discarded from the output (kept only to warm up the encoder), so the
    kept regions exactly tile the full record with no gaps or overlap.
    """
    if n_time <= window_days + context_days:
        return [(0, n_time, 0, n_time)]

    windows = []
    first_end = window_days + context_days
    windows.append((0, first_end, 0, first_end))
    cursor = first_end
    while cursor < n_time:
        w_start = cursor - context_days
        w_end = min(n_time, w_start + context_days + window_days)
        windows.append((w_start, w_end, cursor, w_end))
        cursor = w_end
    return windows


def load_pretrain_dataset(
    nc_path: str,
    pretrained_ts_vars: List[str],
    pretrained_static_vars: List[str],
    start: str,
    end: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load raw (unnormalized) time-series + static arrays for the full
    station set in `nc_path`. Returns (x_ts [T,N,F], c_static [N,S],
    date_range, station_ids).
    """
    nc_tool = NetCDFDataset()
    ts_data, static_data, date_range, station_ids = nc_tool.nc2array(
        nc_path,
        station_ids=None,
        time_range=[start, end],
        time_series_variables=pretrained_ts_vars,
        static_variables=pretrained_static_vars,
        warmup_days=0,
        add_coords=False,
    )
    # nc2array returns [station, time, features] -> encoder/normalize_data
    # convention is time-major [time, station, features].
    x_ts = np.transpose(ts_data, (1, 0, 2)).astype(np.float32)
    c_static = static_data.astype(np.float32)
    return x_ts, c_static, pd.DatetimeIndex(date_range), station_ids


def normalize_pretrain_inputs(
    x_ts: np.ndarray,
    c_static: np.ndarray,
    pretrained_ts_vars: List[str],
    pretrained_static_vars: List[str],
    norm_stats_path: str,
    overwrite: bool,
) -> np.ndarray:
    """Mirrors NnDualLoader's pretrained-stream normalization: a dedicated
    stats file, no log-norm variables, dummy target (there is no downstream
    target here, just the raw variables the encoder was pretrained on).
    """
    dummy_target = np.zeros((x_ts.shape[0], x_ts.shape[1], 1), dtype=np.float32)
    norm_stats = load_norm_stats(
        norm_stats_path,
        overwrite,
        x_ts,
        c_static,
        dummy_target,
        pretrained_ts_vars,
        pretrained_static_vars,
        ["_dummy_"],
        [],
        {},
    )
    return normalize_data(
        x_ts, c_static, pretrained_ts_vars, pretrained_static_vars, norm_stats, [],
    )


def period_codes_for(
    date_range: pd.DatetimeIndex, resolution: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Map each day in date_range to a period index for the given resolution.
    'seasonal' uses meteorological seasons (DJF/MAM/JJA/SON) via pandas' Q-NOV
    quarter convention. Returns (codes [T] in [0, n_periods), period_starts).
    """
    freq = {'monthly': 'M', 'seasonal': 'Q-NOV', 'annual': 'Y'}[resolution]
    period_index = date_range.to_period(freq)
    codes, uniques = pd.factorize(period_index, sort=True)
    period_starts = pd.DatetimeIndex([p.start_time for p in uniques])
    return codes, period_starts


class DailyEmbeddingSink:
    """Streams per-window/basin-batch daily embedding chunks straight to a
    NetCDF file on disk, instead of buffering the full [T, N, D] array in
    memory first. For a multi-decade, multi-thousand-basin record that array
    can be tens of GB -- large enough to OOM a job even when the actual goal
    is the much smaller monthly/seasonal/annual aggregates. Peak memory with
    this sink is bounded by one window's basin-batch slice, regardless of how
    many total days/basins the run covers.
    """

    def __init__(
        self,
        path: str,
        station_ids: np.ndarray,
        date_range: pd.DatetimeIndex,
        embedding_dim: int,
        var_name: str,
        lat: Optional[np.ndarray],
        lon: Optional[np.ndarray],
        complevel: int,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.ds = netCDF4.Dataset(path, 'w', format='NETCDF4')
        self.ds.createDimension('station_ids', len(station_ids))
        self.ds.createDimension('time', len(date_range))
        self.ds.createDimension('embed_dim', embedding_dim)
        self.ds.embedding_dim = int(embedding_dim)

        sid_var = self.ds.createVariable('station_ids', str, ('station_ids',))
        for i, sid in enumerate(station_ids):
            sid_var[i] = str(sid)

        time_var = self.ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'days since 1970-01-01'
        time_var.calendar = 'standard'
        time_var[:] = netCDF4.date2num(
            pd.DatetimeIndex(date_range).to_pydatetime(),
            units=time_var.units, calendar=time_var.calendar,
        )

        chunks = (min(64, len(station_ids)), min(365, len(date_range)), embedding_dim)
        self.emb_var = self.ds.createVariable(
            var_name, 'f4', ('station_ids', 'time', 'embed_dim'),
            zlib=True, complevel=complevel, chunksizes=chunks,
        )

        if lat is not None:
            self.ds.createVariable('lat', 'f8', ('station_ids',))[:] = lat
        if lon is not None:
            self.ds.createVariable('lon', 'f8', ('station_ids',))[:] = lon

    def write(self, t_start: int, t_end: int, b_start: int, b_end: int, kept: np.ndarray) -> None:
        """kept: [kept_len, B, D] time-major slice."""
        self.emb_var[b_start:b_end, t_start:t_end, :] = np.transpose(kept, (1, 0, 2))

    def close(self) -> None:
        self.ds.close()
        size_mb = os.path.getsize(self.path) / (1024 ** 2)
        print(f"Saved {self.path}  ({size_mb:.1f} MB, streamed)")


def encode_and_aggregate(
    model: torch.nn.Module,
    pretrained_ts_vars: List[str],
    pretrained_static_vars: List[str],
    xc_pretrained_norm: np.ndarray,
    date_range: pd.DatetimeIndex,
    d_model: int,
    window_days: int,
    context_days: int,
    batch_size: int,
    device: torch.device,
    agg_resolutions: List[str],
    daily_sink: Optional[DailyEmbeddingSink],
) -> dict:
    """Run the frozen encoder over the whole record via sliding windows and
    basin batches.

    `agg_resolutions` (monthly/seasonal/annual) are accumulated as running
    (sum, count) buckets as each window/basin-batch is encoded -- cheap,
    bounded by the number of periods, not days. If `daily_sink` is given, the
    same per-window output is also streamed straight to disk via it. Neither
    path ever materializes a full [T, N, D] array in memory.

    Returns {resolution: (agg_array, period_starts)} for agg_resolutions.
    """
    T, N, _ = xc_pretrained_norm.shape
    xc_full = torch.from_numpy(xc_pretrained_norm)
    windows = sliding_windows(T, window_days, context_days)

    codes_by_res, starts_by_res, sums_by_res, counts_by_res = {}, {}, {}, {}
    for r in agg_resolutions:
        codes, starts = period_codes_for(date_range, r)
        codes_by_res[r] = codes
        starts_by_res[r] = starts
        sums_by_res[r] = np.zeros((len(starts), N, d_model), dtype=np.float64)
        counts_by_res[r] = np.bincount(codes, minlength=len(starts))

    resolutions_desc = agg_resolutions + (['daily'] if daily_sink is not None else [])
    print(f"Encoding {T} days x {N} basins in {len(windows)} time window(s), "
          f"batch_size={batch_size} basins, resolutions={resolutions_desc}")

    for w_i, (w_start, w_end, keep_start, keep_end) in enumerate(windows):
        temporal_features = extract_temporal_features(date_range[w_start:w_end])
        tf_tensor = torch.from_numpy(temporal_features).float().to(device)
        rel_start, rel_end = keep_start - w_start, keep_end - w_start

        for b0 in range(0, N, batch_size):
            b1 = min(N, b0 + batch_size)
            xc_slice = xc_full[w_start:w_end, b0:b1, :].to(device)

            with torch.no_grad():
                hidden = encode_with_pretrained(
                    model, pretrained_ts_vars, pretrained_static_vars,
                    xc_slice, tf_tensor,
                )  # [B, w_len, d_model]

            hidden_np = hidden.detach().cpu().numpy()
            kept = np.transpose(
                hidden_np[:, rel_start:rel_end, :], (1, 0, 2)
            )  # [kept_len, B, d_model], batch-basin slice only

            if daily_sink is not None:
                daily_sink.write(keep_start, keep_end, b0, b1, kept)

            for r in agg_resolutions:
                codes_slice = codes_by_res[r][keep_start:keep_end]
                np.add.at(sums_by_res[r][:, b0:b1, :], codes_slice, kept)

        print(f"  window {w_i + 1}/{len(windows)}: days [{w_start}:{w_end}) "
              f"-> kept [{keep_start}:{keep_end})")

    aggregated = {}
    for r in agg_resolutions:
        counts = counts_by_res[r][:, None, None]
        aggregated[r] = (
            (sums_by_res[r] / counts).astype(np.float32),
            starts_by_res[r],
        )

    return aggregated


def fetch_lat_lon(
    nc_path: str, station_ids: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort lat/lon lookup for the output file; skipped if absent."""
    try:
        with xr.open_dataset(nc_path) as probe:
            if 'lat' not in probe or 'lon' not in probe or 'station_ids' not in probe:
                return None, None
            probe_ids = probe['station_ids'].values
            id_to_idx = {str(s): i for i, s in enumerate(probe_ids)}
            idx = [id_to_idx.get(str(s)) for s in station_ids]
            if any(i is None for i in idx):
                return None, None
            return probe['lat'].values[idx], probe['lon'].values[idx]
    except Exception:
        return None, None


def save_embedding_nc(
    path: str,
    station_ids: np.ndarray,
    lat: Optional[np.ndarray],
    lon: Optional[np.ndarray],
    dates: pd.DatetimeIndex,
    emb_tnd: np.ndarray,
    var_name: str,
    complevel: int,
) -> None:
    """emb_tnd: [T_or_n_period, N, D] time-major array."""
    data_vars = {
        var_name: (('station_ids', 'time', 'embed_dim'), np.transpose(emb_tnd, (1, 0, 2))),
    }
    if lat is not None:
        data_vars['lat'] = (('station_ids',), lat)
    if lon is not None:
        data_vars['lon'] = (('station_ids',), lon)

    ds = xr.Dataset(data_vars, coords={'station_ids': station_ids, 'time': dates})
    ds.attrs['embedding_dim'] = int(emb_tnd.shape[-1])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path, encoding={var_name: {'zlib': True, 'complevel': complevel}})

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Saved {path}  ({size_mb:.1f} MB, shape {emb_tnd.shape})")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    nn_cfg = OmegaConf.to_container(cfg['model']['nn'], resolve=True)

    pretrained_ts_vars = nn_cfg.get('pretrained_time_series_vars', [])
    pretrained_static_vars = nn_cfg.get('pretrained_static_vars', [])
    if not pretrained_ts_vars or not pretrained_static_vars:
        raise ValueError(
            "--config's model.nn must set 'pretrained_time_series_vars' and "
            "'pretrained_static_vars'."
        )
    pretrained_model_path = nn_cfg.get('pretrained_model')
    if not pretrained_model_path:
        raise ValueError("--config's model.nn must set 'pretrained_model' (checkpoint path).")

    d_model = nn_cfg.get('hidden_size', 256)
    num_heads = nn_cfg.get('num_heads', 4)
    num_enc_layers = nn_cfg.get('num_enc_layers', 4)
    d_ffd = nn_cfg.get('d_ffd', 512)
    dropout = nn_cfg.get('dropout', 0.1)
    pretrained_type = nn_cfg.get('pretrained_type', 'stefaland_dec_lstm')

    pretrain_data = args.pretrain_data or cfg.get('pretrained_path')
    if not pretrain_data:
        raise ValueError("Provide --pretrain_data, or set 'pretrained_path' in --config.")

    out_root = os.path.join(args.out_dir, args.dataset_name)
    os.makedirs(out_root, exist_ok=True)

    if args.start_time is None or args.end_time is None:
        with xr.open_dataset(pretrain_data) as probe:
            file_start, file_end = pd.Timestamp(probe['time'].values.min()), pd.Timestamp(probe['time'].values.max())
        start = args.start_time or file_start.strftime('%Y-%m-%d')
        end = args.end_time or file_end.strftime('%Y-%m-%d')
    else:
        start, end = args.start_time, args.end_time

    print(f"Loading pretrain dataset {pretrain_data} [{start} .. {end}]")
    x_ts, c_static, date_range, station_ids = load_pretrain_dataset(
        pretrain_data, pretrained_ts_vars, pretrained_static_vars, start, end,
    )
    print(f"  {x_ts.shape[1]} basins, {x_ts.shape[0]} days, "
          f"{len(pretrained_ts_vars)} ts vars, {len(pretrained_static_vars)} static vars")

    norm_stats_path = os.path.join(out_root, 'normalization_statistics_pretrained.json')
    xc_pretrained_norm = normalize_pretrain_inputs(
        x_ts, c_static, pretrained_ts_vars, pretrained_static_vars,
        norm_stats_path, args.overwrite_norm_stats,
    )

    print(f"Building frozen encoder (d_model={d_model}, pretrained_type={pretrained_type}) from {pretrained_model_path}")
    model = build_pretrained_encoder(
        d_model, num_heads, dropout, num_enc_layers, d_ffd,
        pretrained_ts_vars, pretrained_static_vars, pretrained_model_path, freeze=True,
        pretrained_type=pretrained_type,
    ).to(device).eval()

    lat, lon = fetch_lat_lon(pretrain_data, station_ids)

    agg_resolutions = [r for r in args.resolutions if r != 'daily']
    daily_sink = None
    if 'daily' in args.resolutions:
        daily_path = os.path.join(out_root, f'{args.dataset_name}_embeddings_daily.nc')
        daily_sink = DailyEmbeddingSink(
            daily_path, station_ids, date_range, d_model,
            args.embedding_var_name, lat, lon, args.complevel,
        )

    t0 = time.time()
    aggregated = encode_and_aggregate(
        model, pretrained_ts_vars, pretrained_static_vars, xc_pretrained_norm,
        date_range, d_model, args.window_days, args.context_days, args.batch_size,
        device, agg_resolutions, daily_sink,
    )
    print(f"Encoding done in {time.time() - t0:.1f}s")

    if daily_sink is not None:
        daily_sink.close()

    for resolution in agg_resolutions:
        emb, dates = aggregated[resolution]
        out_path = os.path.join(out_root, f'{args.dataset_name}_embeddings_{resolution}.nc')
        save_embedding_nc(out_path, station_ids, lat, lon, dates, emb, args.embedding_var_name, args.complevel)


if __name__ == '__main__':
    main()
