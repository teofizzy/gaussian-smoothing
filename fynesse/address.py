"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""
from . import access
from . import assess

from typing import Any, Union
import pandas as pd
import logging
import numpy as np
import xarray as xr
import math
from typing import Tuple, Dict

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}

def compute_correlation_with_smoothing(
    ref_series, other_series, sigmas=(0, 1, 2, 5)
):
    """
    Apply Gaussian smoothing at multiple scales and compute correlations.

    Parameters
    ----------
    ref_series : array-like
        Reference time series.
    other_series : array-like
        Comparison time series.
    sigmas : list or tuple, default=(0, 1, 2, 5)
        Standard deviations for Gaussian smoothing.
        sigma=0 means "no smoothing".

    Returns
    -------
    correlations : dict
        Mapping {sigma: correlation}.
    smoothed_ref : dict
        Mapping {sigma: smoothed ref_series}.
    smoothed_other : dict
        Mapping {sigma: smoothed other_series}.
    """
    ref_series = np.asarray(ref_series, dtype=float)
    other_series = np.asarray(other_series, dtype=float)

    # Drop NaNs at same positions
    mask = ~np.isnan(ref_series) & ~np.isnan(other_series)
    ref_series = ref_series[mask]
    other_series = other_series[mask]

    correlations = {}
    smoothed_ref = {}
    smoothed_other = {}

    for sigma in sigmas:
        if sigma > 0:
            ref_smooth = gaussian_filter1d(ref_series, sigma=sigma)
            other_smooth = gaussian_filter1d(other_series, sigma=sigma)
        else:
            ref_smooth, other_smooth = ref_series, other_series

        smoothed_ref[sigma] = ref_smooth
        smoothed_other[sigma] = other_smooth

        corr = np.nan
        if len(ref_smooth) > 1 and len(other_smooth) > 1:
            corr, _ = pearsonr(ref_smooth, other_smooth)

        correlations[sigma] = corr

    return correlations, smoothed_ref, smoothed_other

def plot_correlation_with_smoothing(
    correlations, smoothed_ref, smoothed_other,
    ref_name="Ref", other_name="Other"
):
    """
    Plot correlation vs sigma and smoothed signal overlays.

    Parameters
    ----------
    correlations : dict
        {sigma: correlation}.
    smoothed_ref : dict
        {sigma: smoothed reference series}.
    smoothed_other : dict
        {sigma: smoothed comparison series}.
    ref_name : str
        Label for reference series.
    other_name : str
        Label for comparison series.
    """
    # Correlation vs sigma
    plt.figure(figsize=(6, 4))
    plt.plot(list(correlations.keys()), list(correlations.values()), marker="o")
    plt.xlabel("Gaussian sigma")
    plt.ylabel("Correlation")
    plt.title(f"Correlation vs. Smoothing ({ref_name} vs {other_name})")
    plt.grid(True)
    plt.show()

    # Smoothed signals
    plt.figure(figsize=(10, 5))
    for sigma, series in smoothed_ref.items():
        plt.plot(series, label=f"{ref_name} (σ={sigma})", alpha=0.7)
    for sigma, series in smoothed_other.items():
        plt.plot(series, "--", label=f"{other_name} (σ={sigma})", alpha=0.7)
    plt.title(f"Smoothed Signals: {ref_name} vs {other_name}")
    plt.legend()
    plt.show()

import seaborn as sns

def analyze_correlations(df):
    """
    Analyze correlation results across stations, datasets, and sigma levels.
    """
    results = {}

    # --- 1. Summary stats per dataset ---
    summary = df.groupby("dataset")[["correlation"]].describe()
    results["summary"] = summary
    print("=== Summary statistics per dataset ===")
    print(summary, "\n")

    # --- 1b. Summary stats per dataset + sigma ---
    summary_sigma = df.groupby(["dataset", "sigma"])[["correlation"]].describe()
    results["summary_sigma"] = summary_sigma
    print("=== Summary statistics per dataset and sigma ===")
    print(summary_sigma, "\n")

    # --- 2. Best/worst stations per dataset ---
    best = df.loc[df.groupby("dataset")["correlation"].idxmax()]
    worst = df.loc[df.groupby("dataset")["correlation"].idxmin()]
    results["best"] = best
    results["worst"] = worst
    print("=== Best station per dataset ===")
    print(best[["dataset", "station_code", "sigma", "correlation"]], "\n")
    print("=== Worst station per dataset ===")
    print(worst[["dataset", "station_code", "sigma", "correlation"]], "\n")

    # --- 3. Distribution plots ---
    plt.figure(figsize=(8,5))
    sns.boxplot(x="dataset", y="correlation", data=df)
    plt.title("Correlation distribution across datasets")
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(x="sigma", y="correlation", hue="dataset", data=df)
    plt.title("Correlation distribution across datasets and sigma")
    plt.legend(title="Dataset")
    plt.show()

    return results

import math

def plot_correlations_df(df, max_cols=2):
    """
    Plot gaussian-filtered correlations for multiple stations and datasets.

    Args:
        df : pd.DataFrame
            Must have columns ['station_code', 'dataset', 'sigma', 'correlation']
    """
    stations = df['station_code'].unique()
    datasets = df['dataset'].unique()

    n_stations = len(stations)

    # ✅ determine grid layout (max 3 columns)
    ncols = min(max_cols, n_stations)
    nrows = math.ceil(n_stations / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)

    for i, station in enumerate(stations):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        df_station = df[df['station_code'] == station]
        for dataset in datasets:
            df_ds = df_station[df_station['dataset'] == dataset]
            df_ds = df_ds.sort_values("sigma")  # sort for consistent plotting
            ax.plot(df_ds['sigma'], df_ds['correlation'], marker='o', label=dataset)

        ax.set_title(f"Station {station}")
        ax.set_xlabel("Gaussian sigma")
        ax.set_ylabel("Correlation with TAHMO")
        ax.grid(True)
        ax.legend()

    # Hide unused subplots if n_stations < nrows*ncols
    for j in range(i+1, nrows*ncols):
        row, col = divmod(j, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.show()

# helper: safe pearson that returns nan if constant / insufficient length
def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0: # check for variability
        return np.nan
    return pearsonr(x[mask], y[mask])[0]

# station diagnostics: missing frac, wet-day frac (>0), mean, std, # nonzero days
def station_diagnostics(series, wet_thresh=0.1):
    """
    Parameters:
        series (time series)
        wet_thresh (float): threshold for wet days in mm
    Returns:
        dict with missing_frac, wet_frac, mean, std, n_wet
    """
    s = np.asarray(series, dtype=float)
    valid = np.isfinite(s)
    n = s.size
    n_valid = valid.sum()
    missing_frac = 1.0 - (n_valid / n)
    if n_valid == 0:
        return {
            "missing_frac": missing_frac,
            "wet_frac": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "n_wet": np.nan
        }
    valid_vals = s[valid]
    n_wet = (valid_vals > wet_thresh).sum()
    wet_frac = n_wet / n_valid
    return {
        "missing_frac": missing_frac,
        "wet_frac": wet_frac,
        "mean": float(np.nanmean(valid_vals)),
        "std": float(np.nanstd(valid_vals, ddof=0)),
        "n_wet": int(n_wet)
    }

# block bootstrap correlation CI for difference of correlations (paired series)
def block_bootstrap_corr_diff(a, b, c, d, block_size=7, n_iter=1000, random_state=0):
    """
    Parameters:
        Block bootstrap to estimate distribution of corr(a,b) - corr(c,d).
        a,b: series for baseline (e.g. sigma=0)
        c,d: series for test (e.g. sigma=5)
    Return: dict with mean_diff, ci_lower, ci_upper, p_two_sided (proportion <=0)
    """
    rng = np.random.default_rng(random_state)
    n = len(a)
    if n != len(b) or n != len(c) or n != len(d):
        raise ValueError("All series must have same length")
    # create block start indices
    block_starts = np.arange(0, n, block_size)
    # list of blocks (start, stop)
    blocks = [(i, min(i+block_size, n)) for i in block_starts]
    mblocks = len(blocks)
    diffs = np.empty(n_iter, dtype=float)
    for it in range(n_iter):
        # sample blocks with replacement until we reach n
        idxs = []
        while len(idxs) < n:
            bidx = rng.integers(0, mblocks)
            s, e = blocks[bidx]
            idxs.extend(range(s, e))
        idxs = idxs[:n]
        a_samp = a[idxs]; b_samp = b[idxs]; c_samp = c[idxs]; d_samp = d[idxs]
        r1 = safe_pearson(a_samp, b_samp)
        r2 = safe_pearson(c_samp, d_samp)
        diffs[it] = (r2 if not np.isnan(r2) else 0.0) - (r1 if not np.isnan(r1) else 0.0)
    mean_diff = np.nanmean(diffs)
    ci_lower, ci_upper = np.nanpercentile(diffs, [2.5, 97.5])
    p_two_sided = np.mean(np.abs(diffs) >= abs(mean_diff))
    return {"mean_diff": mean_diff, "ci": (ci_lower, ci_upper), "p_two_sided": p_two_sided, "diffs": diffs}

def pick_best_worst(df, dataset="CHIRPS", sigma=5, topk=3):
    """
    df: correlations df with columns ['station_code','dataset','sigma','correlation','lat','lon']
    returns (best_codes, worst_codes) lists of station codes (topk each)
    """
    sub = df[(df['dataset'] == dataset) & (df['sigma'] == sigma)].copy()
    if sub.empty:
        raise ValueError("No rows for dataset/sigma")
    sub = sub.dropna(subset=['correlation'])
    sub_sorted = sub.sort_values('correlation', ascending=False)
    best = sub_sorted.head(topk)['station_code'].tolist()
    worst = sub_sorted.tail(topk)['station_code'].tolist()
    return best, worst

def plot_station_pair_time_series(
    station_code,
    metadata_df,
    tahmo_da,
    other_da,
    other_name="CHIRPS",
    sigma=5,
    sig0=0,
    kernel_for_other="temporal",  # 'temporal' means apply gaussian_filter1d to sampled point
    plot_show=True
):
    """
    Plot TAHMO vs other (sampled at station lat/lon) for sigma=sig0 and sigma=sigma.
    Returns dict of diagnostics, correlations, Δr, and spatial distance.
    """
    # lookup coords
    row = metadata_df[metadata_df['code'] == station_code]
    if row.empty:
        raise KeyError(f"station {station_code} not in metadata")
    row = row.iloc[0]
    lat_ref = float(row['location.latitude'])
    lon_ref = float(row['location.longitude'])

    # get raw series (TAHMO)
    tahmo_series = tahmo_da.sel(station=station_code).values.astype(float)

    # sample other at nearest gridpoint
    sel = other_da.sel(lat=lat_ref, lon=lon_ref, method="nearest")
    other_series = sel.values.astype(float)

    # get coordinates of the selected gridpoint (to compute distance)
    lat_other = float(sel.lat.values)
    lon_other = float(sel.lon.values)

    # compute spatial distance
    dist_km = access.haversine_km(lat_ref, lon_ref, lat_other, lon_other)

    # --- smoothing helper ---
    def smooth_1d(series, sigma_val):
        if sigma_val is None or sigma_val == 0:
            return series.copy()
        s = np.asarray(series, dtype=float)
        mask = np.isfinite(s)
        if mask.sum() == 0:
            return np.full_like(s, np.nan)
        s0 = np.where(mask, s, 0.0)
        num = gaussian_filter1d(s0, sigma=sigma_val, mode='nearest')
        w = gaussian_filter1d(mask.astype(float), sigma=sigma_val, mode='nearest')
        out = np.where(w > 1e-6, num / w, np.nan)
        return out

    # --- compute smoothed series ---
    tahmo_s0 = smooth_1d(tahmo_series, sig0)
    tahmo_sS = smooth_1d(tahmo_series, sigma)
    other_s0 = smooth_1d(other_series, sig0)
    other_sS = smooth_1d(other_series, sigma)

    # --- correlations ---
    r_raw = safe_pearson(tahmo_s0, other_s0)
    r_smooth = safe_pearson(tahmo_sS, other_sS)
    delta_r = r_smooth - r_raw

    # --- diagnostics summary ---
    diag = {
        "station_code": station_code,
        "lat_ref": lat_ref,
        "lon_ref": lon_ref,
        "lat_other": lat_other,
        "lon_other": lon_other,
        "distance_km": dist_km,
        "corr_raw": r_raw,
        "corr_sigma": r_smooth,
        "delta_r": delta_r,
        "tahmo_diag_raw": station_diagnostics(tahmo_s0),
        "tahmo_diag_sigma": station_diagnostics(tahmo_sS),
        "other_diag_raw": station_diagnostics(other_s0),
        "other_diag_sigma": station_diagnostics(other_sS),
        "tahmo_series_raw": tahmo_s0,
        "tahmo_series_sigma": tahmo_sS,
        "other_series_raw": other_s0,
        "other_series_sigma": other_sS,
    }

    # --- plotting ---
    if plot_show:
        # Use actual time coordinate instead of numeric index
        if "time" in tahmo_da.dims:
            time_idx = pd.to_datetime(tahmo_da["time"].values)
        else:
            time_idx = np.arange(len(tahmo_series))

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(time_idx, tahmo_s0, label="TAHMO")
        axes[0].plot(time_idx, other_s0, '--', label=f"{other_name} raw")
        axes[0].set_title(
            f"{station_code} raw (σ=0)  r={r_raw:.3f}  dist={dist_km:.1f} km"
        )
        axes[0].legend()

        axes[1].plot(time_idx, tahmo_sS - np.nanmean(tahmo_sS), label=f"TAHMO σ={sigma}")
        axes[1].plot(time_idx, other_sS - np.nanmean(other_sS), '--', label=f"{other_name} σ={sigma}")
        axes[1].set_title(
            f"{station_code} smoothed σ={sigma}  r={r_smooth:.3f}  Δr={delta_r:+.3f}"
        )
        axes[1].legend()

        plt.xlabel("time index")
        plt.tight_layout()
        plt.show()

    return diag

def compare_sigma_levels(df, dataset="CHIRPS", sigma_high=5, sigma_low=0, topk=3):
    """
    Compare correlation changes for best and worst stations between sigma_high and sigma_low.
    Returns a DataFrame summarizing the difference.
    """
    best, worst = pick_best_worst(df, dataset=dataset, sigma=sigma_high, topk=topk)
    target_stations = best + worst

    sub = df[(df['dataset'] == dataset) & (df['station_code'].isin(target_stations))]

    # pivot sigma values into columns for easy comparison
    pivoted = sub.pivot_table(
        index='station_code', columns='sigma', values='correlation'
    ).reset_index()

    pivoted['delta'] = pivoted[sigma_high] - pivoted[sigma_low]
    pivoted['type'] = pivoted['station_code'].apply(
        lambda x: "best" if x in best else "worst"
    )

    return pivoted.sort_values('type')

def inspect_best_worst(
    df,
    metadata_df,
    tahmo_da,
    other_da,
    other_name="CHIRPS",
    sigma=5,
    topk=3,
    sig0=0,
    block_bootstrap_kwargs={"block_size":7, "n_iter":1000, "random_state":0}
):
    best, worst = pick_best_worst(df, dataset=other_name, sigma=sigma, topk=topk)
    picks = {"best": best, "worst": worst}
    results = {}
    for kind, codes in picks.items():
        results[kind] = []
        for code in codes:
            diag = plot_station_pair_time_series(
                code, metadata_df, tahmo_da, other_da,
                other_name=other_name, sigma=sigma, sig0=sig0, plot_show=True
            )
            # bootstrap test: compare raw corr vs sigma corr for this station
            a = diag["tahmo_series_raw"]
            b = diag["other_series_raw"]
            c = diag["tahmo_series_sigma"]
            d = diag["other_series_sigma"]
            bb = block_bootstrap_corr_diff(a, b, c, d, **block_bootstrap_kwargs)
            diag["bootstrap"] = bb
            results[kind].append(diag)
    return results
