"""Assemble the per-year chunks and write the MSWEP variable into a Caravan file.

This is where the time-axis correction happens. Step 02 produced basin means
keyed by true calendar date; the combined Caravan file's `time` coordinate is
fabricated by CombineCamels.py and is wrong by +365 d (camels/camelsbr/camelscl),
+366 d (camelsgb) or +1 d (camelsaus). The underlying values are positionally
intact, so column k of station s really holds the date `first_day[s] + k`.

MSWEP is therefore scattered per station using `first_day[s]` from step 00, which
puts it in register with the ERA5-Land forcings sitting beside it in the same
array. It deliberately does NOT match the file's own `time` labels -- matching
those would misalign MSWEP against every other forcing by up to a year.

Default behaviour copies the source file and appends the new variables to the
copy, leaving the original byte-identical. `--inplace` appends to the original.
"""

import argparse
import csv
import datetime as dt
import os
import shutil
import sys

import netCDF4
import numpy as np
import xarray as xr

import mswep_common as C

DEFAULT_OUT = C.CARAVAN_NC.replace(".nc", "_MSWEP.nc")
EPOCH = dt.date(1900, 1, 1)
OFFSET_VAR = "time_offset_days"


def load_first_days(station_ids):
    """station -> true date of its column 0, as days since 1900."""
    path = f"{C.WORK_DIR}/station_time_axis.csv"
    with open(path) as fh:
        table = {r["station_id"]: int(r["first_day_since_1900"])
                 for r in csv.DictReader(fh)}
    missing = [s for s in station_ids if s not in table]
    if missing:
        raise SystemExit(f"{len(missing)} stations absent from {path}, e.g. {missing[:5]}")
    return np.array([table[s] for s in station_ids], dtype="int64")


def load_chunks(n_station):
    """(day_matrix, day0) where day_matrix[d - day0, station] is the basin mean."""
    names = sorted(f for f in os.listdir(C.CHUNK_DIR)
                   if f.startswith("mswep_") and f.endswith(".npz"))
    if not names:
        raise SystemExit(f"no chunks in {C.CHUNK_DIR}; run step 02 first")

    loaded = []
    for name in names:
        z = np.load(f"{C.CHUNK_DIR}/{name}")
        days, values = z["days_since_1900"], z["values"]
        if values.shape[1] != n_station:
            raise SystemExit(
                f"{name}: {values.shape[1]} basins, expected {n_station}"
            )
        loaded.append((days, values))
        print(f"  {name}: {len(days)} days")

    day0 = int(min(d.min() for d, _ in loaded))
    day1 = int(max(d.max() for d, _ in loaded))
    mat = np.full((day1 - day0 + 1, n_station), np.nan, dtype="float32")
    for days, values in loaded:
        mat[days - day0] = values
    print(f"  MSWEP days {EPOCH + dt.timedelta(days=day0)} .. "
          f"{EPOCH + dt.timedelta(days=day1)} ({mat.shape[0]} rows)")
    return mat, day0


def align(mat, day0, first_days, n_time):
    """Scatter date-keyed MSWEP onto each station's own column axis."""
    n_station = first_days.size
    out = np.full((n_station, n_time), np.nan, dtype="float32")
    k = np.arange(n_time, dtype="int64")

    for first in np.unique(first_days):
        rows = np.where(first_days == first)[0]
        src = first + k - day0
        ok = (src >= 0) & (src < mat.shape[0])
        out[np.ix_(rows, k[ok])] = mat[np.ix_(src[ok], rows)].T
        print(f"  start {EPOCH + dt.timedelta(days=int(first))}: "
              f"{len(rows)} stations, {int(ok.sum())} columns filled")
    return out


def write_variables(path, data, first_days, note):
    label0 = (dt.date(1950, 1, 1) - EPOCH).days
    with netCDF4.Dataset(path, "a") as nc:
        for name in (C.MSWEP_VAR, OFFSET_VAR):
            if name in nc.variables:
                raise SystemExit(
                    f"{path} already has {name}; delete it or pick another output"
                )

        ref = nc.variables["total_precipitation_sum"]
        var = nc.createVariable(
            C.MSWEP_VAR, "f4", ref.dimensions, zlib=True, complevel=4,
            shuffle=True, chunksizes=ref.chunking(), fill_value=np.float32(np.nan),
        )
        var.setncatts({
            "units": "mm/day",
            "long_name": "Catchment-mean daily precipitation from MSWEP",
            "coordinates": "lat lon",
            "source": C.MSWEP_DIR,
            "aggregation": (
                "cos(latitude)-weighted mean of 0.1 deg MSWEP cells over the "
                "Caravan basin polygon, weighted by exact polygon-cell "
                "intersection area; the same spatial reduction Caravan applies "
                "to ERA5-Land for total_precipitation_sum"
            ),
            "time_alignment": (
                "Aligned column-for-column with total_precipitation_sum and the "
                "other ERA5-Land forcings in this file. NOTE: this file's `time` "
                "coordinate is fabricated and does not describe the data; see "
                f"the {OFFSET_VAR} variable for each station's true dates."
            ),
            "comment": note,
        })
        var[:] = data

        off = nc.createVariable(OFFSET_VAR, "i2", (ref.dimensions[0],))
        off.setncatts({
            "long_name": "Days to add to this file's `time` values to get the true date",
            "units": "days",
            "comment": (
                "This file's `time` coordinate was generated as 1950-01-01 + "
                "column index by CombineCamels.py, which discarded the `date` "
                "variable of each source Caravan basin file. The data itself is "
                "positionally identical to the source (verified bit-exact). The "
                "true date of column k for station s is time[k] + "
                f"{OFFSET_VAR}[s]. Offsets: +365 d for camels/camelsbr/camelscl, "
                "+366 d for camelsgb, +1 d for camelsaus. The 561 camelsaus "
                "stations were additionally truncated from 27027 source days to "
                "the first 26662."
            ),
        })
        off[:] = (first_days - label0).astype("i2")


def summarise(mswep, era5):
    """The checks worth reading before trusting the new variable."""
    print("\n--- coverage ---")
    print(f"finite MSWEP values: {100 * np.isfinite(mswep).mean():.3f}%")
    both = np.isfinite(mswep) & np.isfinite(era5)
    print(f"cells with both MSWEP and ERA5-Land: {100 * both.mean():.3f}%")
    if not both.any():
        return

    print("\n--- MSWEP vs Caravan ERA5-Land over their shared cells ---")
    print(f"MSWEP mean {mswep[both].mean():.4f} mm/d, "
          f"ERA5-Land mean {era5[both].mean():.4f} mm/d")

    # Per-basin daily correlation. This is the load-bearing check: a bad polygon
    # match or a residual time offset shows up here as a collapsed correlation.
    n = mswep.shape[0]
    pm, pe, r = (np.full(n, np.nan) for _ in range(3))
    for b in range(n):
        m = both[b]
        if m.sum() < 365:
            continue
        a, c = mswep[b, m].astype("float64"), era5[b, m].astype("float64")
        pm[b], pe[b] = a.mean(), c.mean()
        if a.std() > 0 and c.std() > 0:
            r[b] = np.corrcoef(a, c)[0, 1]

    ok = np.isfinite(r)
    print(f"basins compared: {int(ok.sum())}/{n}")
    print("daily correlation with ERA5-Land:")
    for q in (1, 5, 25, 50, 75, 95):
        print(f"  p{q:<2d} {np.percentile(r[ok], q):.4f}")
    bias = 100 * (pm[ok] - pe[ok]) / pe[ok]
    print("mean-precip bias vs ERA5-Land (%):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<2d} {np.percentile(bias, q):+.2f}")
    low = int((r[ok] < 0.5).sum())
    print(f"basins with r < 0.5: {low}")
    if low:
        idx = np.where(ok)[0][r[ok] < 0.5]
        print(f"  e.g. indices {idx[:10].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--inplace", action="store_true",
                    help="append to the source file instead of copying it")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output file")
    args = ap.parse_args()

    with xr.open_dataset(C.CARAVAN_NC) as ds:
        station_ids = ds["station_ids"].values.astype(str)
        n_time = ds.sizes["time"]
        era5 = ds["total_precipitation_sum"].values

    first_days = load_first_days(station_ids)

    print(f"loading chunks for {len(station_ids)} basins x {n_time} columns ...")
    mat, day0 = load_chunks(len(station_ids))

    print("aligning onto per-station column axes ...")
    data = align(mat, day0, first_days, n_time)

    summarise(data, era5)

    note = (
        f"Built {dt.date.today().isoformat()} from the daily MSWEP archive. "
        "NaN where the station's true dates fall outside the MSWEP record, "
        "which starts 1979-01-01."
    )

    if args.inplace:
        target = C.CARAVAN_NC
        print(f"\nappending {C.MSWEP_VAR} in place -> {target}")
    else:
        target = args.out
        if os.path.exists(target):
            if not args.overwrite:
                raise SystemExit(f"{target} exists; pass --overwrite to replace it")
            os.remove(target)
        print(f"\ncopying source -> {target} ...", flush=True)
        shutil.copyfile(C.CARAVAN_NC, target)

    write_variables(target, data, first_days, note)
    print(f"wrote {C.MSWEP_VAR} and {OFFSET_VAR} to {target}")


if __name__ == "__main__":
    sys.exit(main())
