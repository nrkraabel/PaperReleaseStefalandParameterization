"""Reduce daily MSWEP fields to per-basin means using the step-01 weights.

One SLURM array task per year. Each task walks that year's dates, reads only the
latitude band the basins actually occupy out of each daily MSWEP file, and
applies the sparse weight matrix. Results are checkpointed per year, so a task
that hits the wall clock can simply be resubmitted.

The reduction is
    mean_b = sum_c W[b,c] * P[c] * valid[c]  /  sum_c W[b,c] * valid[c]
i.e. missing cells are dropped and the remaining weights renormalised, with the
day set to NaN if less than MIN_VALID_AREA_FRACTION of the basin has data.

Output here is keyed by *true calendar date*, not by Caravan column index. The
combined Caravan file's `time` coordinate is fabricated and wrong by a different
offset per sub-dataset (see 00_station_time_axis.py), so mapping dates onto
columns is deferred to step 03, which does it per station.
"""

import argparse
import csv
import datetime as dt
import os
import sys
import time

import netCDF4
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

import mswep_common as C

EPOCH = dt.date(1900, 1, 1)


def load_weights():
    """Sparse basin x cell matrix, restricted to the columns actually used.

    Returns (W, used_cols, station_ids, row0, row1) where `used_cols` indexes the
    flattened [row0:row1, :] band and W has one column per used cell.
    """
    z = np.load(C.WEIGHTS_NPZ, allow_pickle=False)
    indptr, cols, data = z["indptr"], z["cols"], z["data"]
    station_ids = z["station_ids"].astype(str)

    rows = cols // C.NLON
    row0, row1 = int(rows.min()), int(rows.max()) + 1

    band_cols = (rows - row0) * C.NLON + (cols % C.NLON)
    used_cols, compact = np.unique(band_cols, return_inverse=True)

    W = csr_matrix(
        (data, compact, indptr), shape=(len(station_ids), used_cols.size)
    )
    return W, used_cols, station_ids, row0, row1


def target_dates():
    """Every true calendar date some station has a column for, that MSWEP covers.

    Each station's column k is really `first_day + k`, and `first_day` varies by
    sub-dataset, so the union of covered dates is wider than any single station's
    span. Intersected with what the MSWEP archive actually holds on disk.
    """
    with xr.open_dataset(C.CARAVAN_NC) as ds:
        n_time = ds.sizes["time"]

    with open(f"{C.WORK_DIR}/station_time_axis.csv") as fh:
        firsts = {int(r["first_day_since_1900"]) for r in csv.DictReader(fh)}
    if not firsts:
        raise SystemExit("station_time_axis.csv is empty; run step 00 first")

    lo = EPOCH + dt.timedelta(days=min(firsts))
    hi = EPOCH + dt.timedelta(days=max(firsts) + n_time - 1)

    days = []
    d = lo
    while d <= hi:
        if os.path.exists(C.mswep_path(d)):
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def read_band(path, row0, row1):
    """Raw float32 MSWEP rows [row0:row1] for one day, undecoded."""
    with netCDF4.Dataset(path) as nc:
        v = nc.variables["precipitation"]
        v.set_auto_maskandscale(False)  # keep the raw fills; see mswep_common
        return np.asarray(v[0, row0:row1, :], dtype="float32")


def run_year(year, W, used_cols, dates, row0, row1, force=False):
    out_path = f"{C.CHUNK_DIR}/mswep_{year:04d}.npz"
    if os.path.exists(out_path) and not force:
        print(f"{year}: already done, skipping")
        return

    days = [d for d in dates if d.year == year]
    if not days:
        print(f"{year}: no dates in this year, nothing to do")
        return

    n_basin = W.shape[0]
    values = np.full((len(days), n_basin), np.nan, dtype="float32")
    kept = []
    n_missing_file = 0
    t0 = time.time()

    for k, day in enumerate(days):
        path = C.mswep_path(day)
        if not os.path.exists(path):
            n_missing_file += 1
            kept.append(day)
            continue

        band = read_band(path, row0, row1).ravel()[used_cols]
        valid = np.isfinite(band) & (band > C.FILL_THRESHOLD)

        num = W @ np.where(valid, band, np.float32(0.0)).astype("float64")
        den = W @ valid.astype("float64")

        row = np.divide(num, den, out=np.full(n_basin, np.nan), where=den > 0)
        row[den < C.MIN_VALID_AREA_FRACTION] = np.nan
        values[k] = row.astype("float32")
        kept.append(day)

        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"  {year}: {k + 1}/{len(days)} days ({rate:.1f} day/s)", flush=True)

    day_num = np.array([(d - EPOCH).days for d in kept], dtype="int64")

    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, days_since_1900=day_num, values=values)
    os.replace(tmp, out_path)  # atomic: a killed task never leaves a partial chunk

    finite = np.isfinite(values)
    print(
        f"{year}: {len(kept)} days, {n_missing_file} MSWEP files absent, "
        f"{100 * finite.mean():.2f}% finite, mean "
        f"{np.nanmean(values) if finite.any() else float('nan'):.3f} mm/d, "
        f"{time.time() - t0:.1f}s -> {out_path}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="single year to process")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo finished years")
    args = ap.parse_args()

    os.makedirs(C.CHUNK_DIR, exist_ok=True)

    W, used_cols, station_ids, row0, row1 = load_weights()
    print(
        f"weights: {W.shape[0]} basins x {used_cols.size} cells "
        f"({W.nnz} nonzeros), MSWEP rows {row0}..{row1}",
        flush=True,
    )

    dates = target_dates()
    have = sorted({d.year for d in dates})
    print(f"dates to cover: {dates[0]} .. {dates[-1]} ({len(dates)} days)", flush=True)

    if args.year is not None:
        years = [args.year]
    else:
        lo = args.start if args.start is not None else have[0]
        hi = args.end if args.end is not None else have[-1]
        years = [y for y in have if lo <= y <= hi]

    for y in years:
        run_year(y, W, used_cols, dates, row0, row1, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
