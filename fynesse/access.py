"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import osmnx as ox
import matplotlib.pyplot as plt
import re
import numpy as np
import cupy as cp
import xarray as xr
from scipy.spatial import cKDTree


# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data(file_path:str) -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv(file_path)

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


def get_osm_datapoints(latitude, longitude, use_km=True, box_size_km=2):
    
    if use_km:
        # Define box width and height
        box_width = box_size_km/111 # 111km = 1 degree at the equator
        box_height = box_size_km/111
    else:
        box_height = 0.02
        box_width = 0.02

    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2

    # Get bounding box
    bbox = (west, south, east, north)
    return bbox

def is_lat_lon(col_name):
    """Checks if a column name is in the 'lat_lon' format."""
    return re.match(r'^-?\d+\.?\d*_-?\d+\.?\d*$', col_name) is not None

def df_to_grid(df):
    """
    df: pd.DataFrame with index = time, columns = 'lat_lon' strings

    Returns:
      arr: np.ndarray shape (ntime, nlat, nlon)
      lats: sorted unique lat array
      lons: sorted unique lon array
      mask: boolean same shape indicating missing values
    """
    cols = df.columns.astype(str)
    coords = np.array([list(map(float, c.split('_'))) for c in cols])
    lats = np.unique(coords[:,0])
    lons = np.unique(coords[:,1])

    #  check grid -- Does not work for tahmo
    if len(lats) * len(lons) != len(cols):
        raise ValueError("Columns do not form a regular lat-lon grid (unique lat * unique lon != ncols).")

    # map col -> index
    lat_idx = {lat:i for i,lat in enumerate(np.sort(lats))}
    lon_idx = {lon:i for i,lon in enumerate(np.sort(lons))}
    ntime = len(df)
    arr = np.full((ntime, len(lat_idx), len(lon_idx)), np.nan, dtype=float)
    for j, col in enumerate(cols):
        lat, lon = coords[j]
        i_lat = lat_idx[lat]
        i_lon = lon_idx[lon]
        arr[:, i_lat, i_lon] = df[col].values
    mask = ~np.isnan(arr)
    return arr, np.sort(lats), np.sort(lons), mask

def df_to_station_array(df):
    """
    Works for irregular station datasets like TAHMO.

    Returns:
      arr: np.ndarray (ntime, nstations)
      coords: np.ndarray (nstations, 2) with lat, lon
      names: list of original column names
    """
    cols = df.columns.astype(str)
    coords = np.array([list(map(float, c.split('_'))) for c in cols])
    arr = df.values  # shape (ntime, nstations)
    return arr, coords, list(cols)


def build_spatial_mapping(ref_lats, ref_lons, other_lats, other_lons):
    """
    For each reference grid point (lat_ref, lon_ref), find nearest (lat_other, lon_other).
    Returns a dict mapping (lat_ref, lon_ref) -> (nearest_lat, nearest_lon).
    """
    # Flatten all points
    ref_points = np.array([(lat, lon) for lat in ref_lats for lon in ref_lons])
    other_points = np.array([(lat, lon) for lat in other_lats for lon in other_lons])

    # KDTree nearest neighbor search
    tree = cKDTree(other_points)
    _, idx = tree.query(ref_points)

    # Map results into dictionary
    nearest_points = other_points[idx]  # shape: (nref, 2)
    spatial_map = {
        (float(lat), float(lon)): (float(nlat), float(nlon))
        for (lat, lon), (nlat, nlon) in zip(ref_points, nearest_points)
    }
    return spatial_map

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points (scalars or arrays)."""
    R = 6371.0
    lat1r = np.radians(lat1); lon1r = np.radians(lon1)
    lat2r = np.radians(lat2); lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
   
try:
    import cupy as cp
except ImportError:
    cp = None
   
def _compute_corr_batched(ref_vals, other_vals, batch_size=2000, use_gpu=False):
    """
    Compute best correlation + index between ref_vals and other_vals in batches.

    Parameters
    ----------
    ref_vals : ndarray (ntime, nref)
        Reference series (time × nref gridpoints/stations).
    other_vals : ndarray (ntime, nother)
        Comparison series (time × nother gridpoints/stations).
    batch_size : int
        Number of reference points to process per batch (default=2000).
    use_gpu : bool
        If True and CuPy is available, compute with GPU.

    Returns
    -------
    best_corr : ndarray (nref,)
        Best correlation value per reference point.
    best_idx : ndarray (nref,)
        Index into `other_vals` giving the location of best_corr.
    """

    xp = cp if (use_gpu and cp is not None) else np  # choose backend

    # --- prep data ---
    ref_vals = np.asarray(ref_vals, dtype=np.float32)
    other_vals = np.asarray(other_vals, dtype=np.float32)

    ntime, nref = ref_vals.shape
    _, nother = other_vals.shape

    best_corr = np.full(nref, np.nan, dtype=np.float32)
    best_idx = np.full(nref, -1, dtype=np.int32)

    # center & nan-safe other
    other_vals = other_vals - np.nanmean(other_vals, axis=0, keepdims=True)
    other_vals = np.nan_to_num(other_vals)

    # move to GPU if requested
    other_xp = xp.asarray(other_vals)
    other_std = xp.linalg.norm(other_xp, axis=0)

    # --- batch loop ---
    for start in range(0, nref, batch_size):
        stop = min(start + batch_size, nref)

        ref_batch = ref_vals[:, start:stop]
        ref_batch = ref_batch - np.nanmean(ref_batch, axis=0, keepdims=True)
        ref_batch = np.nan_to_num(ref_batch)

        ref_xp = xp.asarray(ref_batch)
        ref_std = xp.linalg.norm(ref_xp, axis=0)

        # numerator (batch, nother)
        num = ref_xp.T @ other_xp
        denom = xp.outer(ref_std, other_std)

        valid = denom != 0
        corr = xp.full_like(num, xp.nan, dtype=xp.float32)
        corr[valid] = num[valid] / denom[valid]

        # back to CPU for argmax
        corr_cpu = xp.asnumpy(corr)
        finite_mask = np.isfinite(corr_cpu)
        corr_for_argmax = np.where(finite_mask, corr_cpu, -np.inf)

        best_idx_batch = np.argmax(corr_for_argmax, axis=1)
        best_corr_batch = np.where(
            ~finite_mask.any(axis=1), np.nan, corr_for_argmax.max(axis=1)
        )

        best_corr[start:stop] = best_corr_batch
        best_idx[start:stop] = best_idx_batch

        if xp is cp:
            cp.cuda.Stream.null.synchronize()

    return best_corr, best_idx

def build_conceptual_mapping_full(ref_da, other_da, year=2020,
                                  spatial_fallback=True, fallback_maxdeg=0.5,
                                  station_coords=None):
    """
    Compute a robust conceptual mapping between a set of reference points and 
    candidate points based on correlation, with optional spatial fallback.

    This function:
      1. Computes correlations between time series anomalies of reference points 
         and candidate points (batched + memory-safe).
      2. Selects, for each reference point, the candidate with the highest valid 
         correlation.
      3. Handles rows with all-NaN or invalid correlations gracefully by 
         optionally falling back to nearest spatial neighbor using a KDTree.
      4. Returns both a conceptual mapping (dict) and a best-correlation array 
         aligned to the full reference set.

    Parameters
    ----------
    ref_points_full : list
        Full list of reference point identifiers (station IDs, (lat, lon) tuples, etc.).
    other_points_full : list
        Full list of candidate point identifiers.
    ref_valid_mask : np.ndarray, shape (n_ref_full,)
        Boolean mask indicating which reference points are valid (non-empty time series).
    other_valid_mask : np.ndarray, shape (n_other_full,)
        Boolean mask indicating which candidate points are valid.
    corr_matrix : np.ndarray, shape (n_ref_valid, n_other_valid)
        Precomputed correlation matrix between valid reference and candidate points.
    station_coords_map : dict, optional
        Mapping from station IDs to (lat, lon) coordinates, used if keys are not numeric.
    spatial_fallback : bool, default=True
        Whether to allow nearest-neighbor fallback when correlations are unavailable.
    fallback_maxdeg : float, optional
        Maximum search radius for spatial fallback in degrees (converted internally to km).

    Returns
    -------
    conceptual_map : dict
        Mapping {ref_point -> best_candidate_point or None} for all reference points.
    best_corr_arr : np.ndarray, shape (n_ref_full,)
        Array of best correlation values aligned with ref_points_full, NaN if unavailable.
    meta : dict
        Metadata including:
            - "ref_points_full", "other_points_full"
            - "ref_valid_mask", "other_valid_mask"
            - "ref_index_valid", "other_index_valid"
    """

    # --- helpers ---
    def _make_hashable(key):
        if isinstance(key, np.ndarray):
            if key.shape == ():
                return str(key.item())
            return tuple(key.tolist())
        elif isinstance(key, (list, tuple)):
            return tuple(_make_hashable(k) for k in key)
        else:
            return key

    def _normalize_time_index(arr):
        idx = pd.to_datetime(arr)
        try:
            tz = idx.tz
        except Exception:
            tz = None
        if tz is not None:
            try:
                idx = idx.tz_convert(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
        return idx

    def _extract_point_coords(points):
        out = []
        for p in points:
            if isinstance(p, (tuple, list, np.ndarray)):
                try:
                    out.append((float(p[0]), float(p[1])) if len(p) == 2 else tuple(map(float, p)))
                except Exception:
                    try:
                        if len(p) == 1:
                            out.append(str(p[0]))
                        else:
                            out.append(tuple(map(str, p)))
                    except Exception:
                        out.append(str(p))
            else:
                out.append(str(p))
        return np.asarray(out, dtype=object)

    # ---- station_coords map ----
    station_coords_map = None
    if station_coords is not None:
        station_coords_map = {}
        for k, v in station_coords.items():
            key = str(k)
            if isinstance(v, str) and "_" in v:
                lat_s, lon_s = v.split("_", 1)
                station_coords_map[key] = (float(lat_s), float(lon_s))
            else:
                station_coords_map[key] = (float(v[0]), float(v[1]))

    # ---- normalize time + slice ----
    ref_da = ref_da.assign_coords(time=_normalize_time_index(ref_da.time.values))
    other_da = other_da.assign_coords(time=_normalize_time_index(other_da.time.values))

    ref_year = ref_da.sel(time=str(year))
    other_year = other_da.sel(time=str(year))

    # only align time
    ref_year, other_year = xr.align(ref_year, other_year, join="inner", exclude=["lat", "lon"])

    # ---- flatten into points ----
    if {"lat", "lon"}.issubset(set(ref_year.dims)):
        ref_flat = ref_year.stack(points=("lat", "lon"))
    else:
        ref_flat = ref_year.stack(points=("station",))

    if {"lat", "lon"}.issubset(set(other_year.dims)):
        other_flat = other_year.stack(points=("lat", "lon"))
    else:
        other_flat = other_year.stack(points=("station",))

    ref_points_full = _extract_point_coords(ref_flat.points.values)
    other_points_full = _extract_point_coords(other_flat.points.values)

    ref_vals_full = ref_flat.values
    other_vals_full = other_flat.values

    # ---- validity masks ----
    ref_valid_mask = ~np.isnan(ref_vals_full).all(axis=0)
    other_valid_mask = ~np.isnan(other_vals_full).all(axis=0)

    nref_full = ref_points_full.shape[0]

    if not ref_valid_mask.any() or not other_valid_mask.any():
        conceptual_map = {ref_points_full[i]: None for i in range(nref_full)}
        return conceptual_map, np.full(nref_full, np.nan), {
            "ref_points_full": ref_points_full,
            "other_points_full": other_points_full,
            "ref_valid_mask": ref_valid_mask,
            "other_valid_mask": other_valid_mask,
            "ref_index_valid": np.nonzero(ref_valid_mask)[0],
            "other_index_valid": np.nonzero(other_valid_mask)[0],
        }

    # ---- restrict to valid ----
    ref_vals = ref_vals_full[:, ref_valid_mask]
    other_vals = other_vals_full[:, other_valid_mask]

    # --- compute best correlation/index ---
    best_corr, best_idx = _compute_corr_batched(ref_vals, other_vals, batch_size=2000, use_gpu=True)

    # ---- KDTree fallback ----
    other_numeric_idxs = []
    other_numeric_coords = []
    for j, key in enumerate(other_points_full):
        if isinstance(key, tuple) and len(key) == 2:
            try:
                latf, lonf = float(key[0]), float(key[1])
                other_numeric_idxs.append(j)
                other_numeric_coords.append((latf, lonf))
            except Exception:
                pass
        elif station_coords_map is not None and key in station_coords_map:
            latf, lonf = station_coords_map[key]
            other_numeric_idxs.append(j)
            other_numeric_coords.append((float(latf), float(lonf)))

    if len(other_numeric_coords) > 0:
        other_numeric_coords = np.asarray(other_numeric_coords)
        kd_tree = cKDTree(other_numeric_coords)
        km_thresh = None if fallback_maxdeg is None else float(fallback_maxdeg) * 111.32
    else:
        kd_tree = None
        km_thresh = None

    # ---- map back to full set ----
    conceptual_map = {}
    best_corr_arr = np.full(nref_full, np.nan, dtype=float)

    ref_valid_indices = np.nonzero(ref_valid_mask)[0]
    other_valid_indices = np.nonzero(other_valid_mask)[0]

    for i_full in range(nref_full):
        ref_key = _make_hashable(ref_points_full[i_full])

        if not ref_valid_mask[i_full]:
            conceptual_map[ref_key] = None
            continue

        i_positions = np.nonzero(ref_valid_indices == i_full)[0]
        if len(i_positions) == 0:
            conceptual_map[ref_key] = None
            continue

        i_valid = int(i_positions[0])
        j_valid = int(best_idx_valid[i_valid])
        corr_val = best_corr_valid[i_valid]

        if np.isnan(corr_val):
            mapped = None
            if spatial_fallback and (kd_tree is not None):
                ref_coord = None
                if isinstance(ref_key, tuple) and len(ref_key) == 2:
                    try:
                        ref_coord = (float(ref_key[0]), float(ref_key[1]))
                    except Exception:
                        ref_coord = None
                elif station_coords_map is not None and ref_key in station_coords_map:
                    ref_coord = station_coords_map[ref_key]

                if ref_coord is not None:
                    dist_array, pos = kd_tree.query([ref_coord], k=1)
                    pos = int(pos[0])
                    other_global_idx = int(other_numeric_idxs[pos])
                    other_lat, other_lon = other_numeric_coords[pos]
                    dist_km = haversine_km(ref_coord[0], ref_coord[1], other_lat, other_lon)
                    if (km_thresh is None) or (dist_km <= km_thresh):
                        mapped = other_points_full[other_global_idx]

            conceptual_map[ref_key] = mapped
            best_corr_arr[i_full] = np.nan
            continue

        j_full = int(other_valid_indices[j_valid])
        conceptual_map[ref_key] = other_points_full[j_full]
        best_corr_arr[i_full] = float(corr_val)

    meta = {
        "ref_points_full": ref_points_full,
        "other_points_full": other_points_full,
        "ref_valid_mask": ref_valid_mask,
        "other_valid_mask": other_valid_mask,
        "ref_index_valid": ref_valid_indices,
        "other_index_valid": other_valid_indices,
    }

    return conceptual_map, best_corr_arr, meta

def compare_against_multiple(ref_da, others, year=2020):
    """
    Run both spatial and conceptual mapping for one reference dataset
    against a dictionary of other datasets.

    Parameters
    ----------
    ref_da   : xarray.DataArray (reference, e.g. TAMSAT)
    others   : dict of {name: xarray.DataArray}
    year     : year for conceptual mapping

    Returns
    -------
    results  : dict
        keys = dataset names,
        values = dicts with:
          {
            'spatial': spatial_map,
            'conceptual': {
                'map': conceptual_map,
                'best_corr': best_corr_arr,
                'meta': meta
            }
          }
    """
    results = {}
    for name, other_da in others.items():
        print(f"Processing {name}...")

        # spatial alignment (nearest grid points)
        spatial_map = build_spatial_mapping(
            ref_da.lat.values, ref_da.lon.values,
            other_da.lat.values, other_da.lon.values
        )

        # conceptual alignment (correlation based)
        conceptual_map, best_corr_arr, meta = build_conceptual_mapping_full(
            ref_da, other_da, year=year
        )

        results[name] = {
            "spatial": spatial_map,
            "conceptual": {
                "map": conceptual_map,
                "best_corr": best_corr_arr,
                "meta": meta
            }
        }

    return results

def tahmo_to_xarray(arr, coords, station_dict, time_index, varname="precip", tol=1e-4):
    """
    Convert TAHMO array to xarray.DataArray while skipping stations
    not present in the array.

    Parameters
    ----------
    arr : np.ndarray
        Shape (ntime, nstations)
    coords : np.ndarray
        Shape (nstations, 2), each row = (lat, lon)
    station_dict : dict
        {station_id: "lat_lon"} where lat_lon is "lat_lon" string
    time_index : pandas.DatetimeIndex
    varname : str
        Variable name for DataArray
    tol : float
        Tolerance for float comparison of coords (default 1e-4)

    Returns
    -------
    xarray.DataArray
    """
    ntime, nstations = arr.shape
    assert coords.shape[0] == nstations, "Mismatch in station count vs coords"
    assert len(time_index) == ntime, "Mismatch in time index length"

    # reverse map: (lat,lon) -> station_id
    reverse_map = {}
    for sid, coord_str in station_dict.items():
        lat_str, lon_str = coord_str.split("_")
        lat, lon = float(lat_str), float(lon_str)
        reverse_map[(round(lat,4), round(lon,4))] = sid

    station_ids = []
    lats = []
    lons = []

    for lat, lon in coords:
        key = (round(lat,4), round(lon,4))
        if key in reverse_map:
            station_ids.append(reverse_map[key])
        else:
            # if station not found, assign placeholder
            station_ids.append(f"missing_{lat:.2f}_{lon:.2f}")
        lats.append(lat)
        lons.append(lon)

    da = xr.DataArray(
        arr,
        dims=("time", "station"),
        coords={
            "time": time_index,
            "station": station_ids,
            "lat": ("station", lats),
            "lon": ("station", lons),
        },
        name=varname
    )
    return da

def tahmo_to_grid_like(da):
    """
    Convert station-based TAHMO DataArray into a pseudo-grid
    with stacked (lat, lon) like regular gridded data.
    """
    return da.set_index(station=("lat", "lon"))

def wrap_as_xarray(arr, name, time_index=None, lats=None, lons=None):
    """
    Convert (time, lat, lon) numpy array into an xarray DataArray.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (time, lat, lon).
    name : str
        Variable name.
    time_index : pd.DatetimeIndex, optional
        Datetime index to use for the time dimension.
    lats : np.ndarray, optional
        Array of latitude values of length arr.shape[1].
    lons : np.ndarray, optional
        Array of longitude values of length arr.shape[2].
    """
    t, ny, nx = arr.shape

    # check time
    if time_index is not None:
        assert len(time_index) == t, "Time index length must match array time dimension"
        time = time_index
    else:
        time = np.arange(t)

    # check lats/lons
    if lats is not None:
        assert len(lats) == ny, "Latitude length must match array lat dimension"
    else:
        lats = np.arange(ny)

    if lons is not None:
        assert len(lons) == nx, "Longitude length must match array lon dimension"
    else:
        lons = np.arange(nx)

    return xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        name=name
    )
