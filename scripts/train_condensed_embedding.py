"""
Standalone embedding condenser, with or without streamflow supervision.

Given ONE resolution's precomputed foundation-model embedding file (as
produced by scripts/generate_embeddings.py, or any NetCDF in the same
(station_ids, time, embed_dim) layout), trains a small linear condenser
(embedding_size -> condensed_width). Two modes, picked by which arguments
are given:

  supervised  (--task_nc AND --target_var given)
      total_loss = streamflow_loss + recon_weight * reconstruction_loss
      The task NetCDF supplies a streamflow target for the same station set;
      the embedding is aligned onto the task file's daily calendar.

  recon-only  (neither given)
      total_loss = reconstruction_loss
      No streamflow anywhere -- a plain autoencoder compression of the
      embedding on its own station set and time axis. Use this when the
      stations have no discharge records.

Giving exactly one of --task_nc / --target_var is an error.

Both modes use the FULL set of basins and the FULL time range available --
no PUB/PUR spatial holdout, no dMG NN-model/trainer/Hydra-config machinery.

The reconstruction term keeps the condensed embedding information-
preserving/general; the streamflow term (supervised mode only) keeps it
hydrologically relevant. Both shape the SAME condenser weights
simultaneously via one ordinary combined backward() call.

In supervised mode a lightweight LSTM head (condensed embedding -> daily
streamflow, embedding ONLY, no forcings) supplies the streamflow-loss
signal. Forcings are deliberately withheld from it: if it could see today's
weather directly, it would lean on that shortcut instead of the condensed
embedding, weakening the exact gradient pressure this is for. The head is
NOT a hydrology model and is not saved as a deliverable -- only the
condenser is kept.

The condenser itself never sees streamflow or forcings, only the raw
embedding -- so once trained (in either mode), it can be applied to
embeddings anywhere, including basins with no streamflow observations at
all, via scripts/apply_condenser.py.

Usage
-----
# supervised
python scripts/train_condensed_embedding.py \\
    --embedding_nc /path/to/MyDataset_embeddings_daily.nc \\
    --task_nc /path/to/my_task_data.nc \\
    --target_var streamflow \\
    --condensed_width 64 \\
    --out /path/to/condensers/MyDataset_condenser_daily64.pt \\
    --export_embedding

# recon-only (no streamflow)
python scripts/train_condensed_embedding.py \\
    --embedding_nc /path/to/MyDataset_embeddings_daily.nc \\
    --condensed_width 64 \\
    --out /path/to/condensers/MyDataset_condenser_daily64_recon.pt \\
    --export_embedding
"""

import argparse
import os
import sys
import time
from typing import Tuple

import netCDF4
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Same sys.path setup generate_embeddings.py uses, so `dmg.*` imports resolve
# whether or not the package is pip-installed.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC = os.path.join(_REPO_ROOT, 'src')
_SRC_DMG = os.path.join(_SRC, 'dmg')
for _p in (_SRC, _SRC_DMG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dmg.core.data.loaders.load_nc import NetCDFDataset  # noqa: E402

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from generate_embeddings import save_embedding_nc  # noqa: E402

RECON_WEIGHT_DEFAULT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--embedding_nc', required=True, help="One resolution's precomputed embedding file.")
    p.add_argument('--embedding_var_name', default='embedding')
    p.add_argument(
        '--task_nc', default=None,
        help="Task NetCDF with the streamflow target for the same stations. "
             "Omit (together with --target_var) for recon-only mode.",
    )
    p.add_argument(
        '--target_var', default=None,
        help="Streamflow variable in --task_nc, e.g. QObs or streamflow. "
             "Omit (together with --task_nc) for recon-only mode.",
    )
    p.add_argument(
        '--start_time', default=None,
        help="YYYY-MM-DD. Supervised: subsets the task file (default: its "
             "earliest date). Recon-only: subsets the embedding file's own "
             "time axis (default: its earliest date).",
    )
    p.add_argument(
        '--end_time', default=None,
        help="YYYY-MM-DD. Same semantics as --start_time, at the other end.",
    )
    p.add_argument('--condensed_width', type=int, required=True, help="e.g. 64, or 264 for annual.")
    p.add_argument(
        '--recon_weight', type=float, default=RECON_WEIGHT_DEFAULT,
        help="Supervised mode only: lambda on the reconstruction term, once "
             "both terms are z-scored onto comparable scales (see zscore()). "
             ">1 favors keeping the embedding intact/general; <1 favors "
             "streamflow relevance. Ignored in recon-only mode.",
    )
    p.add_argument('--predictor_hidden_size', type=int, default=64, help="Supervised mode only.")
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64, help="Basins per minibatch.")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help="Output checkpoint path (condenser only).")
    p.add_argument(
        '--export_embedding', action='store_true',
        help="Also write the trained condenser's output over the full "
             "basin/time set as a new <out>_embedding.nc file, in the same "
             "(station_ids, time, embed_dim) format generate_embeddings.py "
             "writes -- a drop-in replacement for the raw embedding file.",
    )
    args = p.parse_args()

    if (args.task_nc is None) != (args.target_var is None):
        p.error(
            "--task_nc and --target_var must be given together (supervised "
            "mode) or both omitted (recon-only mode)."
        )
    args.mode = 'supervised' if args.task_nc is not None else 'recon_only'
    return args


# ============================================================
# Task-side data (target only), full basin set, full time range
# ============================================================
def load_task_data(
    nc_path: str, target_var: str, start: str, end: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Returns (target [T,N,1], date_range, station_ids) for EVERY station in
    the file -- no spatial subsetting of any kind."""
    nc_tool = NetCDFDataset()
    ts_data, _, date_range, station_ids = nc_tool.nc2array(
        nc_path,
        station_ids=None,
        time_range=[start, end],
        time_series_variables=[target_var],
        static_variables=[],
        warmup_days=0,
        add_coords=False,
    )
    if station_ids is None:
        # Caravan/Zenodo-schema files use a bare 'station' dim + 'station_id'
        # data var instead of the 'station_ids' coordinate
        # NetCDFDataset.nc2array looks for, so it comes back None here.
        # Station order is unaffected -- no subsetting happened above since
        # station_ids=None was passed in -- so it's safe to read the same
        # file's station_id var directly, in file order.
        import xarray as xr
        with xr.open_dataset(nc_path) as probe:
            if 'station_id' in probe:
                station_ids = probe['station_id'].values
            else:
                raise ValueError(
                    f"{nc_path} has neither a 'station_ids' coordinate nor a "
                    "'station_id' variable -- can't recover station ids."
                )
    # ts_data: [N, T, 1] -> time-major [T, N, 1]
    target = np.transpose(ts_data, (1, 0, 2)).astype(np.float32)
    return target, pd.DatetimeIndex(date_range), station_ids


def zscore(target: np.ndarray) -> np.ndarray:
    """z-score the target (NaNs stay NaN, still excluded by the loss mask).

    Without this, streamflow_loss (raw-unit MSE -- easily orders of
    magnitude larger than the embedding's own scale) would swamp recon_loss
    regardless of --recon_weight, silently making that knob a no-op. Both
    loss terms need to live on comparable scales for --recon_weight to mean
    what it says.
    """
    mean = np.nanmean(target)
    std = np.nanstd(target)
    std = std if std > 1e-6 else 1.0
    return (target - mean) / std


# ============================================================
# Embedding-side metadata + as-of calendar alignment
# ============================================================
def open_embedding(nc_path: str, var_name: str):
    ds = netCDF4.Dataset(nc_path, 'r')
    emb_ids = np.array([str(s) for s in ds['station_ids'][:]])
    emb_dates = pd.DatetimeIndex(
        netCDF4.num2date(
            ds['time'][:], units=ds['time'].units, calendar=getattr(ds['time'], 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        )
    )
    return ds, ds[var_name], emb_ids, emb_dates


def asof_time_index(emb_dates: pd.DatetimeIndex, task_dates: pd.DatetimeIndex) -> np.ndarray:
    """For each task day, the position in emb_dates (sorted) of the latest
    embedding date <= that day -- forward-fill match, same convention
    EmbeddingFinetuneLoader uses for monthly/seasonal/annual files. Days
    before the embedding file's earliest date map to index 0 (its first,
    least-stale value) rather than being dropped, since every task day here
    needs a row."""
    sort_order = np.argsort(emb_dates.values)
    sorted_dates = emb_dates.values[sort_order]
    pos = np.searchsorted(sorted_dates, task_dates.values, side='right') - 1
    pos = np.clip(pos, 0, len(sorted_dates) - 1)
    return sort_order[pos]


# ============================================================
# Model pieces (plain torch, no dMG NN-model machinery)
# ============================================================
class Condenser(nn.Module):
    def __init__(self, d_in: int, width: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, width)
        self.decoder = nn.Linear(width, d_in)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class StreamflowHead(nn.Module):
    """Lightweight, training-only auxiliary predictor -- not a hydrology
    model, exists purely to supply a streamflow-relevance gradient to the
    condenser in supervised mode. Discarded after training.

    Deliberately takes ONLY the condensed embedding, no forcings: with
    today's weather available directly, this would learn to lean on that
    shortcut instead of the embedding, weakening the pressure on the
    condenser to actually encode streamflow-relevant information.
    """

    def __init__(self, width: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(width, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, condensed):
        # condensed: [T, B, width]
        out, _ = self.lstm(condensed)
        return self.head(out)


# ============================================================
# Data preparation, one function per mode
# ============================================================
def print_cache_size(n_stations: int, n_time: int, d: int) -> None:
    gb = n_stations * n_time * d * 4 / 1024 ** 3
    print(f"Caching {n_stations} x {n_time} x {d} embedding array in memory "
          f"(~{gb:.2f} GB, one-time read) ...")


def read_embedding_rows(emb_var, emb_idx: np.ndarray) -> np.ndarray:
    """Read the given station rows of the embedding variable -> [N, T_emb, D],
    in emb_idx order. netCDF4 fancy indexing needs sorted indices, so sort,
    read, then undo the sort.

    Cached once in memory so the training loop slices this array instead of
    re-decompressing the netCDF file every minibatch of every epoch, which
    dominates runtime at daily resolution."""
    sort_order = np.argsort(emb_idx)
    rows = np.asarray(emb_var[emb_idx[sort_order].tolist(), :, :], dtype=np.float32)
    return rows[np.argsort(sort_order)]


def prepare_supervised(args, emb_var, emb_ids, emb_dates):
    """Returns (raw_cache [N,T,D], target [T,N,1], station_ids, dates), with
    the embedding aligned onto the task file's calendar and station set."""
    start, end = args.start_time, args.end_time
    if start is None or end is None:
        import xarray as xr
        with xr.open_dataset(args.task_nc) as probe:
            start = start or pd.Timestamp(probe['time'].values.min()).strftime('%Y-%m-%d')
            end = end or pd.Timestamp(probe['time'].values.max()).strftime('%Y-%m-%d')

    print(f"Loading task data (ALL basins, {start} .. {end}) from {args.task_nc} ...")
    target, task_dates, task_ids = load_task_data(args.task_nc, args.target_var, start, end)
    print(f"  {target.shape[1]} basins, {target.shape[0]} days, target={args.target_var}")

    emb_id_set = set(emb_ids)
    common_ids = [s for s in (str(x) for x in task_ids) if s in emb_id_set]
    if not common_ids:
        raise ValueError("No overlapping station_ids between --task_nc and --embedding_nc.")
    print(f"  {len(common_ids)}/{len(task_ids)} task basins have a matching embedding")

    task_id_to_idx = {str(s): i for i, s in enumerate(task_ids)}
    emb_id_to_idx = {s: i for i, s in enumerate(emb_ids)}
    task_idx_for_common = np.array([task_id_to_idx[s] for s in common_ids])
    emb_idx_for_common = np.array([emb_id_to_idx[s] for s in common_ids])

    time_idx = asof_time_index(emb_dates, task_dates)  # [T_task] -> position into emb file's time axis
    target_common = zscore(target[:, task_idx_for_common, :])  # [T, N_common, 1]

    print_cache_size(len(common_ids), len(task_dates), emb_var.shape[-1])
    raw_cache = read_embedding_rows(emb_var, emb_idx_for_common)
    raw_cache = raw_cache[:, time_idx, :]  # [N_common, T_task, D], aligned onto task calendar
    return raw_cache, target_common, np.array(common_ids), task_dates


def prepare_recon_only(args, emb_var, emb_ids, emb_dates):
    """Returns (raw_cache [N,T,D], None, station_ids, dates) on the embedding
    file's own station set and time axis -- no task file, no alignment."""
    keep = np.ones(len(emb_dates), dtype=bool)
    if args.start_time is not None:
        keep &= emb_dates >= pd.Timestamp(args.start_time)
    if args.end_time is not None:
        keep &= emb_dates <= pd.Timestamp(args.end_time)
    time_idx = np.nonzero(keep)[0]
    if len(time_idx) == 0:
        raise ValueError("--start_time/--end_time select no timesteps of --embedding_nc.")
    dates = emb_dates[time_idx]
    print(f"  {len(emb_ids)} stations, {len(time_idx)} timesteps "
          f"({dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d})")

    print_cache_size(len(emb_ids), len(time_idx), emb_var.shape[-1])
    raw_cache = read_embedding_rows(emb_var, np.arange(len(emb_ids)))
    raw_cache = raw_cache[:, time_idx, :]
    return raw_cache, None, emb_ids, dates


# ============================================================
# Training
# ============================================================
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    supervised = args.mode == 'supervised'
    print("=" * 70)
    if supervised:
        print(f"MODE: supervised  (streamflow target '{args.target_var}' from {args.task_nc})")
    else:
        print("MODE: recon-only  (no streamflow supervision -- reconstruction loss only)")
        if args.recon_weight != RECON_WEIGHT_DEFAULT:
            print(f"WARNING: --recon_weight={args.recon_weight} is ignored in recon-only mode.")
    print("=" * 70)

    print(f"Opening embedding file {args.embedding_nc} ...")
    emb_ds, emb_var, emb_ids, emb_dates = open_embedding(args.embedding_nc, args.embedding_var_name)
    d_in = emb_var.shape[-1]

    prepare = prepare_supervised if supervised else prepare_recon_only
    raw_cache, target_common, station_ids, dates = prepare(args, emb_var, emb_ids, emb_dates)
    raw_cache = np.nan_to_num(raw_cache, nan=0.0)
    emb_ds.close()
    n_stations = len(station_ids)

    condenser = Condenser(d_in, args.condensed_width).to(device)
    params = list(condenser.parameters())
    predictor = None
    if supervised:
        print(f"Building condenser ({d_in} -> {args.condensed_width}) + streamflow head "
              f"(hidden={args.predictor_hidden_size}) on {device} ...")
        predictor = StreamflowHead(args.condensed_width, args.predictor_hidden_size).to(device)
        params += list(predictor.parameters())
    else:
        print(f"Building condenser ({d_in} -> {args.condensed_width}) on {device} ...")
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mse = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n_stations)
        epoch_stream_loss, epoch_recon_loss, n_batches = 0.0, 0.0, 0

        for b0 in range(0, n_stations, args.batch_size):
            batch_idx = perm[b0:b0 + args.batch_size]

            emb_t = torch.from_numpy(raw_cache[batch_idx]).to(device).permute(1, 0, 2)  # [T, B, D]
            condensed, reconstructed = condenser(emb_t)
            recon_loss = mse(reconstructed, emb_t)

            if supervised:
                target_t = torch.from_numpy(
                    np.ascontiguousarray(target_common[:, batch_idx, :])
                ).to(device)  # [T, B, 1]
                pred = predictor(condensed)
                valid = ~torch.isnan(target_t)
                if valid.any():
                    stream_loss = mse(pred[valid], target_t[valid])
                else:
                    stream_loss = torch.zeros((), device=device)
                total_loss = stream_loss + args.recon_weight * recon_loss
                epoch_stream_loss += stream_loss.item()
            else:
                total_loss = recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            n_batches += 1

        stream_col = f"streamflow_mse={epoch_stream_loss / n_batches:.4f}  " if supervised else ""
        print(f"  epoch {epoch:3d}: {stream_col}recon_mse={epoch_recon_loss / n_batches:.4f}  "
              f"({time.time() - t0:.0f}s elapsed)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        'encoder_state_dict': condenser.encoder.state_dict(),
        'd_in': d_in,
        'width': args.condensed_width,
        'mode': args.mode,
        'source_embedding_nc': args.embedding_nc,
        'task_nc': args.task_nc,
        'target_var': args.target_var,
        'recon_weight': args.recon_weight if supervised else None,
    }, args.out)
    print(f"Saved condenser: {args.out}")

    if args.export_embedding:
        print("Exporting condensed embedding over the full basin/time set ...")
        condenser.eval()
        n_time = raw_cache.shape[1]
        condensed_full = np.zeros((n_stations, n_time, args.condensed_width), dtype=np.float32)
        with torch.no_grad():
            for s0 in range(0, n_stations, args.batch_size):
                idx = np.arange(s0, min(n_stations, s0 + args.batch_size))
                z, _ = condenser(torch.from_numpy(raw_cache[idx]).to(device))
                condensed_full[idx] = z.cpu().numpy()

        emb_out_path = os.path.splitext(args.out)[0] + '_embedding.nc'
        save_embedding_nc(
            emb_out_path,
            station_ids=np.asarray(station_ids),
            lat=None, lon=None,
            dates=dates,
            emb_tnd=np.transpose(condensed_full, (1, 0, 2)),  # [T, N, D] time-major, matching save_embedding_nc's expectation
            var_name='embedding',
            complevel=4,
        )


if __name__ == '__main__':
    main()
