"""Recover the true calendar date of every column in the combined Caravan file.

The combined file's `time` coordinate is fabricated, not read from the data.
CombineCamels.py does:

    n_days = len(ds.dimensions["date"])
    dates  = make_dates_from_dim(n_days)   # 1950-01-01 + 0..n_days-1

i.e. it takes each per-basin Caravan file's *length* and invents a date axis
starting at 1950-01-01, discarding the file's own `date` variable. The values
themselves are copied positionally and are bit-identical to the source, so the
data is fine -- only the labels are wrong, and they are wrong by a different
amount for each sub-dataset:

    camels, camelsbr, camelscl   real axis starts 1951-01-01   (label off by -365 d)
    camelsgb                     real axis starts 1951-01-02   (label off by -366 d)
    camelsaus                    real axis starts 1950-01-02   (label off by -1 d)

camelsaus additionally has 27027 source days against the file's 26662 columns,
so the combiner truncated it to its first 26662 (CombineCamels.py:285).

This step reads every per-basin source file, records its true first date, checks
the axis is contiguous daily, and writes station_time_axis.csv. Everything
downstream aligns MSWEP to these recovered dates so the new variable lines up
with the ERA5-Land forcings sitting beside it in the same array.
"""

import csv
import os
import sys

import netCDF4
import numpy as np
import xarray as xr

import mswep_common as C

TS_ROOT = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "Caravan-nc/timeseries/netcdf"
)
OUT_CSV = f"{C.WORK_DIR}/station_time_axis.csv"


def read_axis(path):
    """(first_day_since_1900, n_days, contiguous) for one per-basin file."""
    with netCDF4.Dataset(path) as nc:
        name = "date" if "date" in nc.variables else "time"
        v = nc.variables[name]
        vals = np.asarray(v[:], dtype="float64")
        units, cal = v.units, getattr(v, "calendar", "standard")

    days = netCDF4.date2num(
        netCDF4.num2date(vals, units, cal), "days since 1900-01-01", "standard"
    )
    days = np.rint(days).astype("int64")
    return int(days[0]), int(days.size), bool(np.all(np.diff(days) == 1))


def main():
    with xr.open_dataset(C.CARAVAN_NC) as ds:
        station_ids = ds["station_ids"].values.astype(str)
        n_time = ds.sizes["time"]

    rows = []
    noncontig, truncated = [], []
    for n, sid in enumerate(station_ids):
        sub = sid.split("_")[0]
        path = f"{TS_ROOT}/{sub}/{sid}.nc"
        if not os.path.exists(path):
            raise SystemExit(f"missing source timeseries for {sid}: {path}")

        first, n_days, contiguous = read_axis(path)
        if not contiguous:
            noncontig.append(sid)
        if n_days != n_time:
            truncated.append((sid, n_days))

        rows.append((sid, sub, first, n_days))
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(station_ids)}", flush=True)

    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["station_id", "sub_dataset", "first_day_since_1900", "n_source_days"])
        wr.writerows(rows)
    print(f"wrote {OUT_CSV}")

    print(f"\nnon-contiguous source axes: {len(noncontig)} {noncontig[:5]}")
    print(f"source longer/shorter than the {n_time} combined columns: {len(truncated)}")

    epoch = np.datetime64("1900-01-01")
    by_sub = {}
    for sid, sub, first, n_days in rows:
        by_sub.setdefault(sub, set()).add((first, n_days))
    print("\ntrue axis by sub-dataset (start, n_source_days) -> label offset:")
    for sub, vals in sorted(by_sub.items()):
        for first, n_days in sorted(vals):
            start = epoch + np.timedelta64(first, "D")
            last = epoch + np.timedelta64(first + n_time - 1, "D")
            off = first - int(
                (np.datetime64("1950-01-01") - epoch) / np.timedelta64(1, "D")
            )
            print(
                f"  {sub:<10} {str(start)} .. {str(last)}  n_src={n_days}  "
                f"label is {off:+d} d off"
            )

    if noncontig:
        raise SystemExit("non-contiguous source axes found; alignment assumption broken")


if __name__ == "__main__":
    sys.exit(main())
