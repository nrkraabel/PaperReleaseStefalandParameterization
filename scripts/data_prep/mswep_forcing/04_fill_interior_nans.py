#!/usr/bin/env python
"""Fill *interior* NaNs in total_precipitation_sum_MSWEP, in place.

Why this is needed
------------------
The merged MSWEP variable has two kinds of NaN:

1. The leading block before each station's 1979 column -- MSWEP simply does
   not start until 1979-01-01. Expected, left alone; the model windows
   (1987-1997 / 1998-2008) sit well after it.
2. Exactly one interior day -- label 2001-12-31, true 2002-12-31 -- NaN for
   284 of the 671 US `camels` basins. That one is a defect.

The interior NaN is not survivable. Two independent paths break on it:

* `loader_utils.calc_stats` uses `np.percentile`/`np.mean`/`np.std`, not the
  nan-aware variants. A single NaN anywhere in the loaded precipitation array
  makes mean/std NaN, `to_norm` then returns all-NaN, and
  `normalize_data`'s `x_nn_norm[x_nn_norm != x_nn_norm] = 0` silently zeroes
  the entire precipitation channel for every basin and every timestep. No
  error, no warning -- just a model trained on zero rain.
* `x_phy` (the HBV [prcp, tmean, pet] stack built in `nn_dual_loader.py` /
  `embedding_finetune_loader.py`) gets no NaN cleaning at all, so the NaN
  propagates through HBV's sequential states to the end of the rho=365
  window and NaNs the loss.

Fix: linear interpolation in time across interior gaps, per basin, only from
each basin's first valid column onward. The leading pre-1979 block keeps its
NaNs -- it is a real absence of record, not a gap to invent data for.

Writes only the affected time columns of the one variable back into the
existing file (netCDF4 in-place slice write), so the 4.8 GB file is not
rewritten. Idempotent: a second run finds nothing to fill.

Usage:
    python 04_fill_interior_nans.py [--dry-run]
"""

import argparse

import numpy as np
from netCDF4 import Dataset

NC_PATH = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids_MSWEP.nc"
)
VAR = "total_precipitation_sum_MSWEP"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=NC_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mode = "r" if args.dry_run else "r+"
    with Dataset(args.nc, mode) as ds:
        var = ds.variables[VAR]
        var.set_auto_mask(False)
        data = var[:]  # (station_ids, time) float32
        n_s, n_t = data.shape

        nan = np.isnan(data)
        # First valid column per basin; basins that are all-NaN are skipped.
        has_valid = nan.any(axis=1) != nan.all(axis=1)
        first_valid = np.full(n_s, n_t, dtype=np.int64)
        any_valid = ~nan.all(axis=1)
        first_valid[any_valid] = np.argmax(~nan[any_valid], axis=1)

        col_idx = np.arange(n_t)[None, :]
        interior = nan & (col_idx >= first_valid[:, None])
        rows = np.where(interior.any(axis=1))[0]

        print(f"{VAR}: {n_s} basins x {n_t} days")
        print(f"  leading (pre-record) NaNs : {int((nan & ~interior).sum())}")
        print(f"  interior NaNs to fill     : {int(interior.sum())} "
              f"across {len(rows)} basins")

        if len(rows) == 0:
            print("  nothing to fill.")
            return

        cols = np.where(interior.any(axis=0))[0]
        print(f"  affected time columns     : {cols.tolist()}")
        for r in rows[:5]:
            bad = np.where(interior[r])[0]
            print(f"    e.g. basin {ds.variables['station_ids'][r]} "
                  f"cols {bad[:10].tolist()}{'...' if len(bad) > 10 else ''}")

        if args.dry_run:
            print("  --dry-run: no write.")
            return

        # Per-basin linear interpolation over the interior gaps only.
        t = np.arange(n_t, dtype=np.float64)
        for r in rows:
            row = data[r]
            good = ~np.isnan(row)
            bad = interior[r]
            row[bad] = np.interp(t[bad], t[good], row[good]).astype(np.float32)
            data[r] = row

        np.clip(data, 0.0, None, out=data, where=~np.isnan(data))

        # Write back only the columns that changed.
        lo, hi = int(cols.min()), int(cols.max()) + 1
        var[:, lo:hi] = data[:, lo:hi]
        ds.sync()

        left = np.isnan(data) & (col_idx >= first_valid[:, None])
        print(f"  wrote columns [{lo}:{hi}); interior NaNs remaining: "
              f"{int(left.sum())}")
        print(f"  filled values: min {np.nanmin(data[rows][:, cols]):.4f} "
              f"max {np.nanmax(data[rows][:, cols]):.4f}")


if __name__ == "__main__":
    main()
