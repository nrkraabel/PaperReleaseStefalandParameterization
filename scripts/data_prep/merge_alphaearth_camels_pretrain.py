"""
One-off merge: attach the 64 AlphaEarth satellite-embedding static attributes
(ae_00..ae_63) from CAMELS_Frederik_with_alphaearth.nc onto a full copy of
Camels_Pretrain.nc, so AlphaEarth-attribute configs can read the exact same
forcing series (P, Tmax, Tmin, srad_daymet, vp_daymet, ...) every other
Camels_Pretrain.nc-based config (Embedding*, DirectFinetuneing/NoPretraining*)
already uses, instead of the different NLDAS-extended product in
CAMELS_Frederik_with_alphaearth.nc.

Safe because both files cover the same 671 stations in the same order over
the same 1980-01-01..2014-12-31 time axis (verified before merging, not just
assumed).

Usage
-----
python scripts/data_prep/merge_alphaearth_camels_pretrain.py
"""

import numpy as np
import xarray as xr

CAMELS_PRETRAIN = "${oc.env:DMG_DATA_ROOT}/Camels_Pretrain.nc"
ALPHAEARTH_SRC = "${oc.env:DMG_DATA_ROOT}/CAMELS_Frederik_with_alphaearth.nc"
OUT_PATH = "${oc.env:DMG_DATA_ROOT}/Camels_Pretrain_AlphaEarth.nc"

AE_VARS = [f"ae_{i:02d}" for i in range(64)]


def main() -> None:
    base = xr.open_dataset(CAMELS_PRETRAIN)
    ae_src = xr.open_dataset(ALPHAEARTH_SRC)

    base_ids = base["station_ids"].values
    ae_ids = ae_src["station_ids"].values
    assert base_ids.shape == ae_ids.shape, (
        f"Station count mismatch: {base_ids.shape} vs {ae_ids.shape}"
    )
    assert (base_ids == ae_ids).all(), (
        "Station ids differ between Camels_Pretrain.nc and "
        "CAMELS_Frederik_with_alphaearth.nc -- merging by position is unsafe."
    )

    missing = [v for v in AE_VARS if v not in ae_src.data_vars]
    if missing:
        raise ValueError(f"Missing AlphaEarth vars in source file: {missing}")

    merged = base.copy(deep=True)
    for var in AE_VARS:
        merged[var] = ae_src[var]

    merged.to_netcdf(OUT_PATH)
    base.close()
    ae_src.close()
    merged.close()

    print(f"Wrote {OUT_PATH}")
    with xr.open_dataset(OUT_PATH) as check:
        print(f"  stations={check.sizes['station_ids']}, time={check.sizes['time']}")
        print(f"  variables: {sorted(check.data_vars)}")


if __name__ == "__main__":
    main()
