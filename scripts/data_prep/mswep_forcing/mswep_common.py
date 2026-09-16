"""Shared paths, grid geometry and constants for the MSWEP -> Caravan pipeline.

The MSWEP daily archive is a global 0.1 deg lat/lon grid, one file per day named
YYYYDDD.nc, holding a single `precipitation` variable in mm/d.

Two things about that archive are load-bearing and not obvious from the header:

  * The declared `_FillValue` is -9999.0 but the value actually stored is
    -239976.0 == -9999 * 24, i.e. the fill survived a per-hour -> per-day unit
    conversion without being re-masked. Anything <= FILL_THRESHOLD is treated as
    missing. (In practice the only fills sit in a 150x150 block north of 75N,
    nowhere near a Caravan basin, but the masking is applied regardless.)

  * The stored lat/lon coordinate vectors are float32 and drift by up to ~1e-4
    deg from the true cell centres. Cell edges are therefore derived
    analytically from the constants below rather than read from the file.
"""

import numpy as np

# ---------------------------------------------------------------- paths

CARAVAN_NC = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids.nc"
)
SHAPEFILE_ROOT = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/"
    "Caravan-nc/shapefiles"
)
MSWEP_DIR = "${oc.env:DMG_DATA_ROOT}/Daily/P_MSWEP"

WORK_DIR = (
    "${oc.env:DMG_DATA_ROOT}/caravan_zenodo/mswep_forcing"
)
WEIGHTS_NPZ = f"{WORK_DIR}/basin_weights.npz"
WEIGHTS_REPORT = f"{WORK_DIR}/basin_weights_report.csv"
CHUNK_DIR = f"{WORK_DIR}/chunks"

# Caravan sub-datasets present in this file (all of the CAMELS family).
SUBDATASETS = ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb"]

# ---------------------------------------------------------------- grid

RES = 0.1
NLAT = 1800
NLON = 3600
LAT_TOP = 90.0
LON_LEFT = -180.0

# Anything at or below this is a fill, not a rainfall total.
FILL_THRESHOLD = -1.0

# Minimum fraction of a basin's area that must have valid MSWEP data for the
# day's basin mean to be emitted; below this the day is NaN.
MIN_VALID_AREA_FRACTION = 0.5

# Name of the variable added to the Caravan file.
MSWEP_VAR = "total_precipitation_sum_MSWEP"


def lat_edges(i):
    """Northern edge of grid row(s) `i`. Row 0 spans 90.0 -> 89.9 degrees."""
    return LAT_TOP - RES * np.asarray(i)


def lon_edges(j):
    """Western edge of grid column(s) `j`. Column 0 spans -180.0 -> -179.9."""
    return LON_LEFT + RES * np.asarray(j)


def lat_centers(i):
    return LAT_TOP - RES * (np.asarray(i) + 0.5)


def mswep_path(date):
    """`datetime.date` -> the MSWEP file for that day (YYYYDDD.nc)."""
    return f"{MSWEP_DIR}/{date.year:04d}{date.timetuple().tm_yday:03d}.nc"
