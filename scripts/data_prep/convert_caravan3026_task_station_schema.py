"""
One-off schema fix for the Caravan3026 task file.

OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue.nc uses the raw
Caravan/Zenodo convention (a bare 'station' dim + 'station_id' data var)
instead of the repo-internal 'station_ids' coordinate that
NetCDFDataset.nc2array (src/dmg/core/data/loaders/load_nc.py) requires --
without this fix, EmbeddingFinetuneLoader/NnDualLoader crash with
`TypeError: 'NoneType' object is not iterable` the same way the condenser
script did before its own fix (see train_condensed_embedding.py's
load_task_data).

This ONLY renames the station dim/coordinate -- no variable values are
changed, dropped, or merged in from any other file (see repo convention:
never cross refpoints/pretrain data into task data).

Streams station-chunks via plain netCDF4 (no dask/xarray-lazy dependency,
no chunks parameter needed) to stay memory-bounded on whatever allocation
this runs under.
"""
import netCDF4
import numpy as np

SRC = "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue.nc"
DST = "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids.nc"
STATION_CHUNK = 200

src = netCDF4.Dataset(SRC, "r")
dst = netCDF4.Dataset(DST, "w", format="NETCDF4")

n_station = src.dimensions["station"].size
n_time = src.dimensions["time"].size
print(f"n_station={n_station} n_time={n_time}")

dst.createDimension("station_ids", n_station)
dst.createDimension("time", n_time)

# time coordinate copied as-is
src_time = src.variables["time"]
dst_time = dst.createVariable("time", src_time.dtype, ("time",))
for attr in src_time.ncattrs():
    if attr == "_FillValue":
        continue
    dst_time.setncattr(attr, src_time.getncattr(attr))
dst_time[:] = src_time[:]

# station_ids coordinate, sourced from the existing station_id data var
station_id_vals = np.asarray(src.variables["station_id"][:]).astype(str)
dst_sid = dst.createVariable("station_ids", str, ("station_ids",))
dst_sid[:] = station_id_vals

# lat/lon must be recognized by xarray as COORDINATES, not plain data vars:
# loader_utils.py's load_nn_data always calls nc2array with add_coords=True,
# and nc2array's ds[selected_vars] subsetting only preserves lat/lon
# automatically if xarray has classified them as coordinates -- otherwise any
# read that doesn't explicitly request them (e.g. NnDualLoader's raw
# HBV-forcing load, which only requests raw_forcing_vars) silently drops
# them, add_coords fails, and HBV physics forcings fall back to a zero-width
# tensor (this is what caused the "Latitude/longitude coordinates not found"
# warning + downstream IndexError on hbv_1_1p_triton.py's
# `x[:, :, self.variables.index('prcp')]`).
#
# Just creating lat/lon as ordinary 1-D netCDF variables is NOT enough --
# xarray only auto-classifies a variable as a coordinate if its name matches
# a dimension name (not the case here: "lat"/"lon" != "station_ids") or if
# some OTHER variable's CF `coordinates` attribute names it. So every 2-D
# (station_ids, time) variable below gets `coordinates = "lat lon"` set,
# which is what actually triggers xarray's auto-promotion on open (verified
# directly: a synthetic file without this attribute drops lat/lon on
# subsetting; with it, they survive). GLOBAL_3434_new.nc apparently has this
# set already, which is why the existing Global3434 configs never hit this.
for coord_name in ("lat", "lon"):
    src_var = src.variables[coord_name]
    vals = np.asarray(src_var[:])
    dst_coord = dst.createVariable(coord_name, src_var.dtype, ("station_ids",))
    dst_coord[:] = vals

for name, var in src.variables.items():
    if name in ("time", "station_id", "lat", "lon"):
        continue
    if var.dimensions == ("station",):
        new_dims = ("station_ids",)
    elif var.dimensions == ("station", "time"):
        new_dims = ("station_ids", "time")
    else:
        print(f"skipping unexpected-dims var {name}: {var.dimensions}")
        continue

    is_numpy_dtype = isinstance(var.dtype, np.dtype)
    kwargs = {}
    if is_numpy_dtype and var.dtype.kind in "fc":
        kwargs["zlib"] = True
        kwargs["complevel"] = 4
        kwargs["fill_value"] = np.nan
    dst_var = dst.createVariable(name, var.dtype if is_numpy_dtype else str, new_dims, **kwargs)
    for attr in var.ncattrs():
        if attr == "_FillValue":
            continue
        dst_var.setncattr(attr, var.getncattr(attr))
    if len(new_dims) == 2:
        dst_var.setncattr("coordinates", "lat lon")

    if len(new_dims) == 1:
        dst_var[:] = var[:]
    else:
        for start in range(0, n_station, STATION_CHUNK):
            end = min(start + STATION_CHUNK, n_station)
            dst_var[start:end, :] = var[start:end, :]
        print(f"copied {name}")

src.close()
dst.close()
print(f"Wrote {DST}")
