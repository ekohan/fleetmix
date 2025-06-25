"""Utilities for converting between Cluster objects and DataFrames.

This module provides conversion functions to transform between the native
Cluster dataclass representation and pandas DataFrames. This allows the
public API to expose clean Python objects while internal implementations
can still leverage pandas for vectorized operations where beneficial.
"""

import numpy as np
import pandas as pd

from fleetmix.core_types import Cluster


def clusters_to_dataframe(clusters: list[Cluster]) -> pd.DataFrame:
    """Convert list of Cluster objects to DataFrame.
    
    Args:
        clusters: List of Cluster objects to convert
        
    Returns:
        DataFrame with columns matching the Cluster dataclass fields
    """
    if len(clusters) == 0:
        return pd.DataFrame()

    data = []
    for cluster in clusters:
        row = {
            "Cluster_ID": cluster.cluster_id,
            "Config_ID": cluster.config_id,
            "Customers": cluster.customers,
            "Total_Demand": cluster.total_demand,
            "Centroid_Latitude": cluster.centroid_latitude,
            "Centroid_Longitude": cluster.centroid_longitude,
            "Goods_In_Config": cluster.goods_in_config,
            "Route_Time": cluster.route_time,
            "Method": cluster.method,
        }
        # Only add TSP_Sequence if it exists and is not empty
        if cluster.tsp_sequence:
            row["TSP_Sequence"] = cluster.tsp_sequence
        data.append(row)

    return pd.DataFrame(data)


def dataframe_to_clusters(df: pd.DataFrame) -> list[Cluster]:
    """Convert DataFrame to list of Cluster objects.
    
    Args:
        df: DataFrame with cluster data
        
    Returns:
        List of Cluster objects
    """
    clusters = []
    for _, row in df.iterrows():
        # Handle TSP_Sequence column if it exists
        tsp_sequence = []
        if "TSP_Sequence" in df.columns and row.get("TSP_Sequence") is not None:
            value = row.get("TSP_Sequence")
            if pd.notna(value) if not isinstance(value, (list, np.ndarray)) else len(value) > 0:
                tsp_sequence = (
                    row["TSP_Sequence"] if isinstance(row["TSP_Sequence"], list) else []
                )

        # Extract goods_in_config if available, otherwise derive from Total_Demand
        goods_in_config = []
        if "Goods_In_Config" in df.columns and row.get("Goods_In_Config") is not None:
            value = row.get("Goods_In_Config")
            if pd.notna(value) if not isinstance(value, (list, np.ndarray)) else len(value) > 0:
                goods_in_config = (
                    row["Goods_In_Config"]
                    if isinstance(row["Goods_In_Config"], list)
                    else []
                )
        elif "Total_Demand" in df.columns and isinstance(row.get("Total_Demand"), dict):
            goods_in_config = [
                good for good, demand in row["Total_Demand"].items() if demand > 0
            ]

        cluster = Cluster(
            cluster_id=row["Cluster_ID"],
            config_id=row.get(
                "Config_ID", "unassigned"
            ),  # Handle missing Config_ID
            customers=row["Customers"]
            if isinstance(row.get("Customers"), list)
            else [],
            total_demand=row["Total_Demand"]
            if isinstance(row.get("Total_Demand"), dict)
            else {},
            centroid_latitude=row.get(
                "Centroid_Latitude", 0.0
            ),  # Handle missing centroids
            centroid_longitude=row.get(
                "Centroid_Longitude", 0.0
            ),  # Handle missing centroids
            goods_in_config=goods_in_config,
            route_time=row.get("Route_Time", 0.0),
            method=row.get("Method", ""),
            tsp_sequence=tsp_sequence,
        )
        clusters.append(cluster)
    return clusters 
