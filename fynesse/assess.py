from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access
import osmnx as ox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    from osmnx.features import InsufficientResponseError

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
