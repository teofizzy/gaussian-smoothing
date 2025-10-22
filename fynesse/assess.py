from typing import Any, Union
import pandas as pd
import logging

from . import access
import osmnx as ox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import sample
import numpy as np
from . import address

import os
import json
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from tqdm.notebook import tqdm


# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def get_osm_features(bbox, place_name, tags):
    pois = ox.features_from_bbox(bbox, tags)
    area = None
    if place_name:
        area = ox.geocode_to_gdf(place_name)
    graph = ox.graph_from_bbox(bbox)
    nodes, edges = ox.graph_to_gdfs(graph)

    return pois, area, nodes, edges

def get_pois_df(pois):
    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
    return pois_df

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Given a central point (latitude, longitude) and a bounding box size,
    query OpenStreetMap via OSMnx and return a feature vector.

    Parameters
    ----------
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size : float
        Size of the bounding box in kilometers
    features : list of tuples
        List of (key, value) pairs to count. Example:
        [
            ("amenity", None),
            ("amenity", "school"),
            ("shop", None),
            ("tourism", "hotel"),
        ]

    Returns
    -------
    feature_vector : dict
        Dictionary of feature counts, keyed by (key, value).
    """
    from ox.features import InsufficientResponseError

    bbox = access.get_osm_datapoints(latitude, longitude)

    # Query OSMnx for features
    tags = {}
    for feature in features:
        tags[feature[0]] = True
    try:
      pois = get_osm_features(bbox, None, tags)[0]
    except InsufficientResponseError:
      return {}

    # Count features matching each (key, value) in poi_types
    pois_df = get_pois_df(pois)

    # Return dictionary of counts
    poi_counts = {}

    for key, value in features:
        if key in pois_df.columns:
            if value:  # count only that value
                poi_counts[f"{key}:{value}"] = (pois_df[key] == value).sum()
            else:  # count any non-null entry
                poi_counts[key] = pois_df[key].notnull().sum()
        else:
            poi_counts[f"{key}:{value}" if value else key] = 0

    return poi_counts

def build_feature_dataframe(city_dicts, features, box_size_km=1):
    results = {}
    for country, cities in city_dicts:
        for city, coords in cities.items():
            vec = get_feature_vector(
                coords["latitude"],
                coords["longitude"],
                box_size_km=box_size_km,
                features=features
            )
            vec["country"] = country
            results[city] = vec
    return pd.DataFrame(results).T

def visualize_feature_space(df, X, label1, label2, label1_color, label2_color):
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    for label, color in [(label1, label1_color), (label2, label2_color)]:
        mask = (y == label)
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1],
                    label=label, color=color, s=100, alpha=0.7)

    for i, feature in enumerate(df.index):
        plt.text(X_proj[i,0]+0.02, X_proj[i,1], feature, fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D projection of feature vectors")
    plt.legend()
    plt.show()

def plot_city_map(city, country, latitude=None, longitude=None, box_size=2, plot_from_place=False):
    """
    Plot a map of a city with points of interest.
    """
    tags = {
    "amenity": True,
    "buildings": True,
    "historic": True,
    "leisure": True,
    "shop": True,
    "tourism": True,
    "religion": True,
    "memorial": True
    }
    place_name = f"{city}, {country}"

    if latitude or longitude:
        # Get bbox
        bbox = access.get_osm_datapoints(latitude, longitude)
        pois = ox.features_from_bbox(bbox, tags=tags)

        # Get graph elements
        graph = ox.graph_from_bbox(bbox)
        area = ox.geocode_to_gdf(place_name)
        nodes, edges = ox.graph_to_gdfs(graph)
        buildings = ox.features_from_bbox(bbox, tags={"building": True})

    if plot_from_place:
        graph = ox.graph_from_place(place_name, network_type='all')
        pois = ox.features_from_place(place_name, tags=tags)
        area = ox.geocode_to_gdf(place_name)
        nodes, edges = ox.graph_to_gdfs(graph)
        buildings = ox.features_from_place(place_name, tags={"building": True})

    # Plot the city map
    fig, ax = plt.subplots(figsize=(6,6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    pois.plot(ax=ax, color="green", markersize=5, alpha=1)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(place_name, fontsize=14)
    plt.show()

def compare_correlation_against_many(ref_series, others_dict, sigmas=[0, 1, 2, 5], plot=False):
    """
    Compare correlations of one reference series vs many others,
    with optional Gaussian smoothing.

    Parameters
    ----------
    ref_series : array-like or pd.Series
        Reference time series (e.g., TAHMO station).
    others_dict : dict[str, array-like or pd.Series]
        Dictionary of {name: time_series} to compare against.
    sigmas : list of int/float
        Gaussian kernel widths to test. 0 means no smoothing.
    plot : bool
        If True, plots correlation vs sigma for each dataset.

    Returns
    -------
    results : dict
        {name: {sigma: correlation_value}}
    """
    results = {}
    for name, series in others_dict.items():
        print(f"Comparing {name}...")
        results[name] = address.compute_correlation_with_smoothing(
            ref_series, series, sigmas=sigmas
            )
        if plot:
            address.plot_correlation_with_smoothing(
                results[name][0],
                results[name][1],
                results[name][2],
                ref_name="TAHMO",
                other_name=name
            )
    return results

from random import sample

def compute_station_correlations_df(
    metadata_df,
    tahmo_da, chirps_da, tamsat_da, era5_da,
    sigmas=[0, 1, 2, 5, 10, 20],
    use_all_stations=False,
    n_random=None,
    random_state=42
):
    """
    Compute gaussian-filtered correlations between TAHMO and other datasets.

    Args:
        metadata_df (pd.DataFrame): Contains station metadata with 'code', 'location.latitude', 'location.longitude'
        tahmo_da (xr.DataArray): TAHMO data with dimension 'station'
        chirps_da, tamsat_da, era5_da (xr.DataArray): Gridded datasets with lat/lon
        sigmas (list): Gaussian filter widths
        use_all_stations (bool): If True, compute for all stations in metadata_df
        n_random (int): If set, randomly select N stations instead of all
        random_state (int): Random seed

    Returns:
        pd.DataFrame with columns:
            ['station_code', 'lat', 'lon', 'dataset', 'sigma', 'correlation']
    """
    # --- station selection logic ---
    stations = list(metadata_df["code"].values)

    if use_all_stations:
        selected_stations = stations
    elif n_random is not None:
        rng = np.random.RandomState(random_state)
        selected_stations = rng.choice(stations, size=n_random, replace=False)
    else:
        raise ValueError("Must set either use_all_stations=True or provide n_random.")

    all_results = []

    # --- main loop ---
    for station_code in selected_stations:
        row = metadata_df[metadata_df['code'] == station_code].iloc[0]
        lat, lon = row['location.latitude'], row['location.longitude']

        try:
            # extract aligned series
            tahmo_series = tahmo_da.sel(station=station_code).values
            chirps_series = chirps_da.sel(lat=lat, lon=lon, method="nearest").values
            tamsat_series = tamsat_da.sel(lat=lat, lon=lon, method="nearest").values
            era5_series = era5_da.sel(lat=lat, lon=lon, method="nearest").values

            others = {
                "CHIRPS": chirps_series,
                "ERA5": era5_series,
                "TAMSAT": tamsat_series,
            }

            # compute correlations for each dataset and sigma
            for dataset_name, other_series in others.items():
                correlations, _, _ = address.compute_correlation_with_smoothing(
                    tahmo_series, other_series, sigmas=sigmas
                )
                for sigma, corr in correlations.items():
                    all_results.append({
                        "station_code": station_code,
                        "lat": lat,
                        "lon": lon,
                        "dataset": dataset_name,
                        "sigma": sigma,
                        "correlation": corr
                    })

            print(f"Processed station {station_code} successfully.")

        except Exception as e:
            print(f"Skipping station {station_code} due to error: {e}")

    return pd.DataFrame(all_results)


def safe_pearson(a, b):
    """NaN-safe Pearson correlation."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two points (in km)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def cross_corr_with_lag(a, b, max_lag=7, use_causal=False):
    """
    Compute cross-correlation and find the lag with maximum correlation.
    Returns (best_corr, best_lag)
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, np.nan

    a = a[mask]
    b = b[mask]
    a -= np.nanmean(a)
    b -= np.nanmean(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan

    corr = correlate(b, a, mode="full", method="auto")
    lags = np.arange(-len(a) + 1, len(a))

    if use_causal:
        # zero out future contributions to enforce causal correlation
        corr[lags < 0] = np.nan

    lag_mask = np.abs(lags) <= max_lag
    corr = corr[lag_mask]
    lags = lags[lag_mask]

    if corr.size == 0:
        return np.nan, np.nan

    best_idx = np.nanargmax(corr)
    norm = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
    best_corr = corr[best_idx] / norm
    best_lag = lags[best_idx]
    return float(best_corr), int(best_lag)


def smooth_1d(series, sigma_val):
    """Gaussian smoothing with NaN handling (non-causal)."""
    if sigma_val is None or sigma_val == 0:
        return series.copy()
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() == 0:
        return np.full_like(s, np.nan)
    s0 = np.where(mask, s, 0.0)
    num = gaussian_filter1d(s0, sigma=sigma_val, mode="nearest")
    w = gaussian_filter1d(mask.astype(float), sigma=sigma_val, mode="nearest")
    return np.where(w > 1e-6, num / w, np.nan)


def smooth_1d_causal(series, sigma_val):
    """Causal Gaussian smoothing (past-to-present only)."""
    if sigma_val is None or sigma_val == 0:
        return series.copy()
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() == 0:
        return np.full_like(s, np.nan)
    s0 = np.where(mask, s, 0.0)
    
    # manual causal kernel
    # Half width = 3 * sigma val, includes 99.7 % of the Gaussian's total area
    # since 3 * sigma covers almost all weight
    half_width = int(3 * sigma_val) # How far back in time the smoother looks (in samples)
    kernel = np.exp(-0.5 * (np.arange(0, half_width + 1) / sigma_val) ** 2)
    kernel /= kernel.sum()
    out = np.full_like(s0, np.nan)
    for i in range(len(s0)):
        i0 = max(0, i - half_width)
        w = kernel[-(i - i0 + 1):]
        seg = s0[i0:i + 1]
        m = mask[i0:i + 1]
        if m.sum() == 0:
            continue
        out[i] = np.sum(seg[m] * w[m]) / np.sum(w[m])
    return out


# -----------------------------------------------
# === Main diagnostics ===
# -----------------------------------------------

def diagnostics_for_station(
    station_code,
    metadata_df,
    tahmo_da,
    other_das_dict,
    sigmas=(0, 5),
    max_lag=7,
    use_causal=False,
    save_dir=None
):
    """Compute correlation and lag diagnostics for one station."""

    if not isinstance(other_das_dict, dict):
        raise TypeError(f"Expected dict for other_das_dict, got {type(other_das_dict)}")

    row = metadata_df.loc[metadata_df["code"] == station_code]
    if row.empty:
        raise KeyError(f"Station {station_code} not found in metadata")
    row = row.iloc[0]
    lat_ref, lon_ref = float(row["location.latitude"]), float(row["location.longitude"])

    tahmo_series = tahmo_da.sel(station=station_code).values.astype(float)
    diagnostics = []

    for name, other_da in other_das_dict.items():
        sel = other_da.sel(lat=lat_ref, lon=lon_ref, method="nearest")
        other_series = sel.values.astype(float)
        lat_other, lon_other = float(sel.lat.values), float(sel.lon.values)
        dist_km = haversine_km(lat_ref, lon_ref, lat_other, lon_other)

        entry = {
            "dataset": name,
            "station_code": station_code,
            "lat_ref": lat_ref,
            "lon_ref": lon_ref,
            "lat_other": lat_other,
            "lon_other": lon_other,
            "distance_km": dist_km,
        }

        for sigma in sigmas:
            if use_causal:
                tahmo_s = smooth_1d_causal(tahmo_series, sigma)
                other_s = smooth_1d_causal(other_series, sigma)
            else:
                tahmo_s = smooth_1d(tahmo_series, sigma)
                other_s = smooth_1d(other_series, sigma)

            corr, lag = cross_corr_with_lag(
                tahmo_s, other_s, max_lag=max_lag, use_causal=use_causal
            )
            entry[f"corr_sigma_{sigma}"] = corr
            entry[f"lag_sigma_{sigma}"] = lag

        diagnostics.append(entry)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{station_code}_diagnostics.json")
        with open(out_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    return diagnostics


# -----------------------------------------------
# === Batch run for all stations ===
# -----------------------------------------------

def run_all_station_diagnostics(
    metadata_df,
    tahmo_da,
    other_das_dict,
    sigmas=(0, 5),
    max_lag=7,
    use_causal=False,
    station_list=None,
    save_dir=None
):
    """Run diagnostics across all stations."""
    if not isinstance(other_das_dict, dict):
        raise TypeError(f"Expected dict for other_das_dict, got {type(other_das_dict)}")

    if station_list is None:
        station_list = metadata_df["code"].tolist()

    all_results = []
    for st in tqdm(station_list, desc="Running station diagnostics"):
        try:
            diag = diagnostics_for_station(
                st,
                metadata_df,
                tahmo_da,
                other_das_dict,
                sigmas=sigmas,
                max_lag=max_lag,
                use_causal=use_causal,
                save_dir=save_dir,
            )
            all_results.extend(diag)
        except Exception as e:
            print(f"⚠️ Skipping {st}: {e}")
            continue

    if save_dir:
        all_json = os.path.join(save_dir, "all_station_diagnostics.json")
        with open(all_json, "w") as f:
            json.dump(all_results, f, indent=2)

    return pd.DataFrame(all_results)

def smooth_centered(series, sigma):
    """Centered Gaussian smoothing (using SciPy) with NaN handling."""
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() == 0 or sigma <= 0:
        return series.copy()
    s0 = np.where(mask, s, 0.0)
    num = gaussian_filter1d(s0, sigma=sigma, mode='nearest')
    w = gaussian_filter1d(mask.astype(float), sigma=sigma, mode='nearest')
    out = np.where(w > 1e-6, num / w, np.nan)
    return out

def smooth_causal(series, sigma, truncate=4.0):
    """
    One-sided (causal) Gaussian-like smoothing: kernel uses past & present.
    """
    if sigma <= 0:
        return series.copy()
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s).astype(float)
    s0 = np.where(mask > 0, s, 0.0)

    radius = int(truncate * sigma + 0.5)
    idx = np.arange(-radius, radius+1)
    gauss = np.exp(-0.5 * (idx / float(sigma))**2)
    gauss_causal = np.where(idx <= 0, gauss, 0.0)
    if gauss_causal.sum() == 0:
        return series.copy()
    kernel = gauss_causal / gauss_causal.sum()

    num = fftconvolve(s0, kernel, mode='same')
    denom = fftconvolve(mask, kernel, mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(denom > 1e-8, num / denom, np.nan)
    return out

def smooth_causal_padded(series, sigma, truncate=4.0):
    if sigma <= 0:
        return series.copy()
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s).astype(float)
    s0 = np.where(mask > 0, s, 0.0)

    radius = int(truncate * sigma + 0.5)
    idx = np.arange(-radius, radius+1)
    gauss = np.exp(-0.5 * (idx / sigma)**2)
    gauss_causal = np.where(idx <= 0, gauss, 0.0)
    kernel = gauss_causal / gauss_causal.sum()

    # pad on both sides with zeros to prevent wraparound
    pad = len(kernel)
    s_pad = np.pad(s0, (pad, pad), mode='constant', constant_values=0)
    m_pad = np.pad(mask, (pad, pad), mode='constant', constant_values=0)

    num = fftconvolve(s_pad, kernel, mode='same')[pad:-pad]
    denom = fftconvolve(m_pad, kernel, mode='same')[pad:-pad]

    out = np.where(denom > 1e-8, num / denom, np.nan)
    return out
