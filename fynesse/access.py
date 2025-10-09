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
from tqdm import tqdm


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

   
def _compute_corr_batched(ref_vals, other_vals, batch_size=2000, use_gpu=True, show_progress=True):
    """
    Batched best-correlation finder between reference time-series and candidate time-series.

    For each reference column (ntime,), find the index of the other column with the highest
    Pearson correlation (computed using centered series). Works with NumPy (CPU) or CuPy (GPU).

    Parameters
    ----------
    ref_vals : ndarray, shape (ntime, nref)
        Time × reference points.
    other_vals : ndarray, shape (ntime, nother)
        Time × candidate points.
    batch_size : int
        Number of reference columns processed per batch.
    use_gpu : bool
        If True and CuPy is available, run heavy math on GPU.
    show_progress : bool
        Show tqdm progress bar for batches.

    Returns
    -------
    best_corr : ndarray, shape (nref,)
      Best correlation value per reference column (np.nan if none valid).
    best_idx : ndarray, shape (nref,)
      Index (0..nother-1) of best candidate per reference column (-1 if none).
    """
    xp = cp if (use_gpu and cp is not None) else np

    # ensure float32 in CPU memory first (we convert to xp arrays inside)
    ref_vals = np.asarray(ref_vals, dtype=np.float32)
    other_vals = np.asarray(other_vals, dtype=np.float32)

    ntime, nref = ref_vals.shape
    _, nother = other_vals.shape

    best_corr = np.full(nref, np.nan, dtype=np.float32)
    best_idx = np.full(nref, -1, dtype=np.int32)

    # Move other_vals to backend xp and center there (nan-safe)
    other_xp = xp.asarray(other_vals)                 # cupy or numpy array on correct backend
    # use xp.nanmean if available; cupy has nanmean in modern versions
    try:
        other_mean = xp.nanmean(other_xp, axis=0, keepdims=True)
    except AttributeError:
        # fallback: compute on CPU and move to xp
        other_mean = xp.asarray(np.nanmean(other_vals, axis=0, keepdims=True))
    other_xp = other_xp - other_mean
    other_xp = xp.nan_to_num(other_xp)                # replace NaNs with 0
    other_std = xp.linalg.norm(other_xp, axis=0)      # xp.linalg.norm (cupy or numpy)

    batch_count = (nref + batch_size - 1) // batch_size
    iterator = range(0, nref, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=batch_count, desc="Batches")

    for start in iterator:
        stop = min(start + batch_size, nref)
        ref_batch = ref_vals[:, start:stop]            # still a NumPy array on host

        # move ref batch to xp and center (nan-safe) on same backend
        ref_xp = xp.asarray(ref_batch)
        try:
            ref_mean = xp.nanmean(ref_xp, axis=0, keepdims=True)
        except AttributeError:
            ref_mean = xp.asarray(np.nanmean(ref_batch, axis=0, keepdims=True))
        ref_xp = ref_xp - ref_mean
        ref_xp = xp.nan_to_num(ref_xp)
        ref_std = xp.linalg.norm(ref_xp, axis=0)      # shape (batch,)

        # numerator and denominator on xp
        # num shape: (batch, nother)
        num = ref_xp.T @ other_xp                      # uses xp matmul
        denom = xp.outer(ref_std, other_std)           # xp.outer

        # compute correlation safely
        valid = denom != 0
        corr = xp.full_like(num, xp.nan, dtype=xp.float32)
        corr[valid] = num[valid] / denom[valid]

        # move to CPU for argmax/argmax-with-masked -inf
        corr_cpu = cp.asnumpy(corr) if xp is cp else corr
        finite_mask = np.isfinite(corr_cpu)
        corr_for_argmax = np.where(finite_mask, corr_cpu, -np.inf)

        best_idx_batch = np.argmax(corr_for_argmax, axis=1)
        best_corr_batch = np.where(~finite_mask.any(axis=1), np.nan, corr_for_argmax.max(axis=1))

        best_corr[start:stop] = best_corr_batch
        best_idx[start:stop] = best_idx_batch

        if xp is cp:
            # ensure GPU work completed
            cp.cuda.Stream.null.synchronize()

    return best_corr, best_idx

def build_conceptual_mapping_full(ref_da, other_da, year=2020,
                                  spatial_fallback=True, fallback_maxdeg=0.5,
                                  station_coords=None, use_gpu=True,
                                  corr_batch_size=2000):
    """
    Compute a robust conceptual mapping between a set of reference points and
    candidate points based on correlation (batched), with optional spatial fallback.

    Returns
    -------
    conceptual_map : dict
        Mapping {ref_key -> other_key or None} for all reference points in the full ref grid.
        Keys are hashable (tuples or station IDs as strings).
    best_corr_arr : numpy.ndarray, shape (n_ref_full,)
        Best correlation value per reference point (NaN if none).
    meta : dict
        Diagnostic metadata:
            - ref_points_full, other_points_full
            - ref_valid_mask, other_valid_mask
            - ref_index_valid, other_index_valid
    Parameters
    ----------
    ref_da, other_da : xarray.DataArray
      Must have time coord and either dims ('time','lat','lon') or ('time','station').
    year : int
      Year used to slice the data (ref_da.sel(time=str(year))).
    spatial_fallback : bool
      If True, attempt spatial fallback when correlations are invalid.
    fallback_maxdeg : float or None
      Maximum allowed distance in degrees for fallback (None => unlimited).
    station_coords : dict or None
      Mapping station_id -> (lat, lon) or 'lat_lon' string for station fallback.
    use_gpu : bool
      Whether to try GPU (CuPy). Falls back to CPU automatically if Cupy not found.
    corr_batch_size : int
      Batch size for correlation computation (tune to fit memory / GPU).
    """
    # -------- helpers --------
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
    try:
        import cupy as cp
    except ImportError:
        cp = None
    try:
        xp = cp if (use_gpu and cp is not None) else np
    except Exception as e:
        print(f"⚠️ GPU backend failed ({e}), using CPU fallback.")
        xp = np
        use_gpu = False

    # -------- station coords normalization --------
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

    # -------- normalize time + select year (only align time) --------
    ref_da = ref_da.assign_coords(time=_normalize_time_index(ref_da.time.values))
    other_da = other_da.assign_coords(time=_normalize_time_index(other_da.time.values))

    ref_year = ref_da.sel(time=str(year))
    other_year = other_da.sel(time=str(year))

    # align only on time (so lat/lon remain intact)
    ref_year, other_year = xr.align(ref_year, other_year, join="inner", exclude=["lat", "lon"])

    # -------- flatten into points --------
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

    ref_vals_full = ref_flat.values    # (ntime, nref_full)
    other_vals_full = other_flat.values

    # ---- validity masks (not all-NaN across time) ----
    ref_valid_mask = ~np.isnan(ref_vals_full).all(axis=0)
    other_valid_mask = ~np.isnan(other_vals_full).all(axis=0)

    nref_full = ref_points_full.shape[0]

    # quick exit if nothing valid
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

    # ---- restrict to valid columns for batched correlation ----
    ref_vals = ref_vals_full[:, ref_valid_mask]       # (ntime, nref_valid)
    other_vals = other_vals_full[:, other_valid_mask] # (ntime, nother_valid)

    # ---- compute best correlation (batched) ----
    # returns numpy arrays: best_corr_valid (nref_valid,), best_idx_valid (nref_valid,)
    best_corr_valid, best_idx_valid = _compute_corr_batched(
        ref_vals, other_vals, batch_size=corr_batch_size, use_gpu=use_gpu
    )

    # ---- build KDTree on numeric OTHER points but only for valid others ----
    other_numeric_idxs = []
    other_numeric_coords = []
    for j_full, key in enumerate(other_points_full):
        if not other_valid_mask[j_full]:
            continue  # only include valid other points
        # key is numeric lat/lon tuple?
        if isinstance(key, tuple) and len(key) == 2:
            try:
                latf, lonf = float(key[0]), float(key[1])
                other_numeric_idxs.append(j_full)
                other_numeric_coords.append((latf, lonf))
            except Exception:
                pass
        else:
            # station-id -> use station_coords_map if available
            if station_coords_map is not None and key in station_coords_map:
                latf, lonf = station_coords_map[key]
                other_numeric_idxs.append(j_full)
                other_numeric_coords.append((float(latf), float(lonf)))

    if len(other_numeric_coords) > 0:
        other_numeric_coords = np.asarray(other_numeric_coords)
        kd_tree = cKDTree(other_numeric_coords)
        km_thresh = None if fallback_maxdeg is None else float(fallback_maxdeg) * 111.32
    else:
        kd_tree = None
        km_thresh = None

    # ---- map results back to FULL ref set ----
    conceptual_map = {}
    best_corr_arr = np.full(nref_full, np.nan, dtype=float)

    ref_valid_indices = np.nonzero(ref_valid_mask)[0]   # maps reduced idx -> global idx
    other_valid_indices = np.nonzero(other_valid_mask)[0]  # maps reduced other idx -> global idx

    # For speed, build a small lookup for mapping global ref index -> reduced index
    map_full_to_valid = np.full(nref_full, -1, dtype=int)
    map_full_to_valid[ref_valid_indices] = np.arange(ref_valid_indices.size, dtype=int)

    for i_full in range(nref_full):
        ref_key = _make_hashable(ref_points_full[i_full])

        if not ref_valid_mask[i_full]:
            conceptual_map[ref_key] = None
            continue

        i_valid = map_full_to_valid[i_full]  # index into reduced ref arrays
        if i_valid < 0:
            conceptual_map[ref_key] = None
            continue

        # get best candidate index in reduced-other space, and corr value
        j_valid = int(best_idx_valid[i_valid])
        corr_val = float(best_corr_valid[i_valid]) if not np.isnan(best_corr_valid[i_valid]) else np.nan

        # if correlation not available or argmax pointed to an invalid index (-1), use spatial fallback
        if np.isnan(corr_val) or (j_valid < 0):
            mapped = None
            if spatial_fallback and (kd_tree is not None):
                # compute numeric coords for reference (from ref_points_full or station_coords_map)
                ref_coord = None
                if isinstance(ref_points_full[i_full], tuple) and len(ref_points_full[i_full]) == 2:
                    try:
                        ref_coord = (float(ref_points_full[i_full][0]), float(ref_points_full[i_full][1]))
                    except Exception:
                        ref_coord = None
                else:
                    # station-id?
                    if station_coords_map is not None and ref_key in station_coords_map:
                        ref_coord = station_coords_map[ref_key]

                if ref_coord is not None:
                    # find nearest candidate in other_numeric_coords (which are global indices stored in other_numeric_idxs)
                    dist_array, pos = kd_tree.query([ref_coord], k=1)
                    pos = int(pos[0])
                    other_global_idx = int(other_numeric_idxs[pos])
                    other_lat, other_lon = other_numeric_coords[pos]
                    dist_km = haversine_km(ref_coord[0], ref_coord[1], other_lat, other_lon)
                    if (km_thresh is None) or (dist_km <= km_thresh):
                        mapped = other_points_full[other_global_idx]
                    else:
                        mapped = None
                else:
                    mapped = None
            conceptual_map[ref_key] = mapped
            best_corr_arr[i_full] = np.nan
            continue

        # map reduced other index back to full other index and then get point id
        j_full = int(other_valid_indices[j_valid])
        conceptual_map[ref_key] = other_points_full[j_full]
        best_corr_arr[i_full] = float(corr_val)

    # ---- meta info ----
    meta = {
        "ref_points_full": ref_points_full,
        "other_points_full": other_points_full,
        "ref_valid_mask": ref_valid_mask,
        "other_valid_mask": other_valid_mask,
        "ref_index_valid": ref_valid_indices,
        "other_index_valid": other_valid_indices
    }

    return conceptual_map, best_corr_arr, meta

def compare_against_multiple(ref_da, others, year=2020, use_gpu=False):
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
        print(f"→ Processing {name} for {year} (GPU={use_gpu})")

        # spatial alignment (nearest grid points)
        spatial_map = build_spatial_mapping(
            ref_da.lat.values, ref_da.lon.values,
            other_da.lat.values, other_da.lon.values
        )

        # conceptual alignment (correlation based)
        conceptual_map, best_corr_arr, meta = build_conceptual_mapping_full(
            ref_da=ref_da,
            other_da=other_da,
            year=year,
            use_gpu=use_gpu,  # ← propagate the flag
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
