"""
Attach the 64 AlphaEarth satellite-embedding static attributes (ae_00..ae_63)
to the Caravan3026 task file, by station_id join.

This is the one prerequisite for
conf/Caravan3026/LSTMHBVAlphaEarthPUB_{PureSpatial,SpatialTemporal}.yaml.

Why a join instead of a fresh Earth Engine sampling
---------------------------------------------------
Caravan_global_singlefile_with_alphaearth.nc already carries ae_00..ae_63 for
all 16299 Caravan basins, sampled from GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
averaged over 2017-2024 (see $DMG_DATA_ROOT/AlphaEarthEmbeddings.py). Every one
of the 3026 station_ids in the Caravan3026 task file
('camels_01013500'-style) is present there exactly once, with zero NaNs --
verified, not assumed, and re-asserted at run time below.

That makes this join safe in the way the SM/GWR-4477 case was not (see
scripts/extract_alphaearth_smgwr4477.py, which had to sample EE directly
because 4477 stations collapsed onto only 978 distinct Caravan basins, one of
them covering 846 stations -- copying basin-level embeddings down would have
leaked spatial information across a PUB holdout). Here the mapping is 1:1
basin-to-basin, so each Caravan3026 station gets its own genuinely distinct
AlphaEarth vector and the PUB split stays honest. It also means this script
needs no internet and no Earth Engine login, unlike the 4477 one.

How it writes
-------------
The task file is 4.7 GB, so it is copied on disk and the 64 new variables
(3026 float32 each, ~775 KB total) are appended to the copy in place via
netCDF4 append mode. Reading it into xarray and calling to_netcdf() would
rewrite all 4.7 GB through memory for the sake of 775 KB of new data -- the
same reason scripts/convert_caravan3026_task_station_schema.py streams rather
than round-trips.

Usage
-----
Run through Slurm, not on a login node (a 4.7 GB copy trips the login-node
cgroup cap the same way the schema conversion did):

    sbatch scripts/data_prep/merge_alphaearth_caravan3026.sh
"""

import argparse
import os
import shutil

import netCDF4
import numpy as np

TASK_NC = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids.nc"
)
ALPHAEARTH_SRC = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "Caravan_global_singlefile_with_alphaearth.nc"
)
OUT_NC = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids_alphaearth.nc"
)

N_AE = 64
AE_VARS = [f"ae_{i:02d}" for i in range(N_AE)]
STATION_DIM = "station_ids"


def gather_embeddings(task_ids: np.ndarray, src_path: str) -> np.ndarray:
    """AlphaEarth matrix (n_task_stations, 64) reordered onto task_ids."""
    with netCDF4.Dataset(src_path) as src:
        src_ids = np.asarray(src.variables["station_id"][:]).astype(str)

        if len(set(src_ids)) != len(src_ids):
            raise ValueError(
                f"{src_path} has duplicate station_id values -- the join would "
                "be ambiguous."
            )

        pos = {sid: i for i, sid in enumerate(src_ids)}
        missing = [sid for sid in task_ids if sid not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} Caravan3026 stations absent from the AlphaEarth "
                f"source (first few: {missing[:5]}). A partial merge would leave "
                "NaN static attributes, which the configs read with no NaN "
                "handling."
            )

        idx = np.array([pos[sid] for sid in task_ids])
        missing_vars = [v for v in AE_VARS if v not in src.variables]
        if missing_vars:
            raise ValueError(f"Missing AlphaEarth vars in source: {missing_vars}")

        ae = np.empty((len(task_ids), N_AE), dtype=np.float32)
        for i, var in enumerate(AE_VARS):
            ae[:, i] = np.asarray(src.variables[var][:])[idx]

    n_nan = int(np.isnan(ae).any(axis=1).sum())
    if n_nan:
        raise ValueError(
            f"{n_nan}/{len(task_ids)} stations carry >=1 NaN AlphaEarth value. "
            "The configs treat these as plain static attributes with no NaN "
            "handling, so a partial result would silently poison training."
        )
    return ae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_nc", default=TASK_NC)
    parser.add_argument("--alphaearth_nc", default=ALPHAEARTH_SRC)
    parser.add_argument("--out_nc", default=OUT_NC)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the output file already exists.",
    )
    args = parser.parse_args()

    if os.path.exists(args.out_nc) and not args.overwrite:
        raise SystemExit(f"{args.out_nc} already exists (pass --overwrite to rebuild)")

    with netCDF4.Dataset(args.task_nc) as task:
        task_ids = np.asarray(task.variables[STATION_DIM][:]).astype(str)
    print(f"[task] {len(task_ids)} stations, e.g. {task_ids[:3].tolist()}")

    ae = gather_embeddings(task_ids, args.alphaearth_nc)
    print(f"[ae]   matrix {ae.shape}, range [{ae.min():.4f}, {ae.max():.4f}], no NaNs")

    print(f"[copy] {args.task_nc}\n    -> {args.out_nc}")
    shutil.copyfile(args.task_nc, args.out_nc)

    with netCDF4.Dataset(args.out_nc, "a") as out:
        for i, var in enumerate(AE_VARS):
            if var in out.variables:
                raise ValueError(f"{var} already present in {args.out_nc}")
            v = out.createVariable(var, "f4", (STATION_DIM,))
            v[:] = ae[:, i]
            v.long_name = f"AlphaEarth embedding dim {i}"
            v.years_averaged = "2017-2024"
            v.source = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
            v.merged_from = os.path.basename(args.alphaearth_nc)
            v.merge_key = "station_id (1:1 Caravan basin match)"

    with netCDF4.Dataset(args.out_nc) as check:
        got = [v for v in AE_VARS if v in check.variables]
        sample = np.asarray(check.variables["ae_00"][:])
        print(f"Wrote {args.out_nc}")
        print(f"  stations={check.dimensions[STATION_DIM].size}, "
              f"time={check.dimensions['time'].size}")
        print(f"  added {len(got)} AlphaEarth vars: {got[0]} ... {got[-1]}")
        print(f"  ae_00 matches source: {np.array_equal(sample, ae[:, 0])}")


if __name__ == "__main__":
    main()
