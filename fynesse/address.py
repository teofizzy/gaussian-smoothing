"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
import pandas as pd
import logging
import numpy as np

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
