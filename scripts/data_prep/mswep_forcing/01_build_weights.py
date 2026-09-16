"""Build the sparse basin x MSWEP-cell area-weight matrix.

Caravan's `total_precipitation_sum` is an area-weighted mean of ERA5-Land over
the catchment polygon, not a point sample at the gauge. To make the MSWEP series
a drop-in replacement it has to be aggregated the same way, so this step
computes, for every Caravan basin, the fraction of the basin's area that falls
in each 0.1 deg MSWEP cell.

Weights are exact polygon-cell intersection areas in degrees^2, scaled by
cos(latitude) of the cell centre so that cells are weighted by true surface area
rather than by their footprint in degrees. Each basin's weights are normalised to
sum to 1, which makes the daily reduction a plain sparse mat-vec and lets the
extract step renormalise cheaply when a cell is missing.

Output: basin_weights.npz (CSR-style arrays, station order identical to the
Caravan file) plus a per-basin report CSV for sanity-checking coverage.
"""

import csv
import sys

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

import mswep_common as C

# 1 degree of latitude in km; used only for the area cross-check in the report.
KM_PER_DEG = 111.32


def load_polygons():
    """gauge_id -> geometry for every CAMELS-family Caravan sub-dataset."""
    geoms = {}
    for name in C.SUBDATASETS:
        path = f"{C.SHAPEFILE_ROOT}/{name}/{name}_basin_shapes.shp"
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            raise SystemExit(f"{path}: expected EPSG:4326, got {gdf.crs}")

        # Several of the shapefiles carry self-intersecting rings (203 in
        # camelsaus and camelscl, 41 in camelsbr). Repair them: an invalid
        # geometry silently returns empty intersections.
        invalid = ~gdf.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = shapely.make_valid(
                gdf.loc[invalid, "geometry"].values
            )
            print(f"  {name}: repaired {int(invalid.sum())} invalid geometries")

        for gid, geom in zip(gdf["gauge_id"].astype(str), gdf.geometry.values):
            geoms[gid] = geom
        print(f"  {name}: {len(gdf)} polygons")
    return geoms


def basin_weights(geom):
    """Return (rows, cols, weights) of MSWEP cells overlapping one polygon.

    `rows`/`cols` are grid indices; weights are cos(lat)-scaled intersection
    areas, not yet normalised.
    """
    minx, miny, maxx, maxy = geom.bounds

    j0 = max(int(np.floor((minx - C.LON_LEFT) / C.RES)), 0)
    j1 = min(int(np.ceil((maxx - C.LON_LEFT) / C.RES)), C.NLON)
    i0 = max(int(np.floor((C.LAT_TOP - maxy) / C.RES)), 0)
    i1 = min(int(np.ceil((C.LAT_TOP - miny) / C.RES)), C.NLAT)
    if j1 <= j0 or i1 <= i0:
        return None

    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()

    north = C.lat_edges(ii)
    west = C.lon_edges(jj)
    cells = shapely.box(west, north - C.RES, west + C.RES, north)

    # Prepared-geometry fast path: cells strictly inside the basin contribute a
    # full cell and never need an intersection computed. For the large Brazilian
    # catchments that is the overwhelming majority of the 60k+ candidate cells.
    shapely.prepare(geom)
    inside = shapely.contains_properly(geom, cells)

    area = np.zeros(cells.shape, dtype="float64")
    area[inside] = C.RES * C.RES

    edge = np.where(~inside)[0]
    if edge.size:
        hits = edge[shapely.intersects(geom, cells[edge])]
        if hits.size:
            area[hits] = shapely.area(shapely.intersection(cells[hits], geom))

    keep = area > 0.0
    if not keep.any():
        # Basin smaller than the numerical noise floor of the intersection, or a
        # degenerate sliver. Fall back to the single cell holding a point that is
        # guaranteed to lie inside the polygon.
        pt = geom.representative_point()
        i = int(np.clip((C.LAT_TOP - pt.y) / C.RES, 0, C.NLAT - 1))
        j = int(np.clip((pt.x - C.LON_LEFT) / C.RES, 0, C.NLON - 1))
        return np.array([i]), np.array([j]), np.array([1.0])

    ii, jj, area = ii[keep], jj[keep], area[keep]
    w = area * np.cos(np.deg2rad(C.lat_centers(ii)))
    return ii, jj, w


def main():
    print("reading Caravan station order ...")
    with xr.open_dataset(C.CARAVAN_NC) as ds:
        station_ids = ds["station_ids"].values.astype(str)
        caravan_area = ds["area"].values.astype("float64")
    print(f"  {len(station_ids)} stations")

    print("loading basin polygons ...")
    geoms = load_polygons()

    missing = [s for s in station_ids if s not in geoms]
    if missing:
        raise SystemExit(f"{len(missing)} stations have no polygon, e.g. {missing[:5]}")

    indptr = np.zeros(len(station_ids) + 1, dtype="int64")
    cols, data = [], []
    report = []

    for n, sid in enumerate(station_ids):
        out = basin_weights(geoms[sid])
        if out is None:
            raise SystemExit(f"{sid}: polygon bounds fall outside the MSWEP grid")
        ii, jj, w = out

        raw_km2 = w.sum() * KM_PER_DEG * KM_PER_DEG
        w = w / w.sum()

        cols.append(ii.astype("int64") * C.NLON + jj.astype("int64"))
        data.append(w)
        indptr[n + 1] = indptr[n] + w.size

        report.append(
            (sid, w.size, round(raw_km2, 3), round(float(caravan_area[n]), 3))
        )
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(station_ids)} basins, {indptr[n + 1]} weights")

    cols = np.concatenate(cols)
    data = np.concatenate(data)

    np.savez_compressed(
        C.WEIGHTS_NPZ,
        indptr=indptr,
        cols=cols,
        data=data,
        station_ids=station_ids,
        nlat=C.NLAT,
        nlon=C.NLON,
        res=C.RES,
    )
    print(f"\nwrote {C.WEIGHTS_NPZ}: {len(station_ids)} basins, {data.size} weights")

    with open(C.WEIGHTS_REPORT, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["gauge_id", "n_cells", "polygon_area_km2", "caravan_area_km2"])
        wr.writerows(report)
    print(f"wrote {C.WEIGHTS_REPORT}")

    # Coverage cross-check: the cos-lat weighted polygon area should track the
    # `area` attribute Caravan ships. Large disagreements mean the wrong polygon
    # got matched to a gauge, which would silently poison that basin's series.
    poly = np.array([r[2] for r in report])
    ref = np.array([r[3] for r in report])
    ok = np.isfinite(ref) & (ref > 0)
    ratio = poly[ok] / ref[ok]
    print("\narea ratio (polygon / Caravan `area`):")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:<2d} {np.percentile(ratio, q):.4f}")
    bad = np.where(ok)[0][(ratio < 0.5) | (ratio > 2.0)]
    print(f"  basins off by >2x: {bad.size}")
    for n in bad[:10]:
        print(f"    {station_ids[n]}: poly {poly[n]:.1f} vs caravan {ref[n]:.1f} km2")

    lat_rows = cols // C.NLON
    print(f"\nrow span touched: {lat_rows.min()}..{lat_rows.max()} of {C.NLAT}")


if __name__ == "__main__":
    sys.exit(main())
