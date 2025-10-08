"""Tests for cluster conversion utilities."""

import pandas as pd

from fleetmix.core_types import Cluster
from fleetmix.utils import cluster_conversion


def test_clusters_to_dataframe_preserves_fields():
    clusters = [
        Cluster(
            cluster_id="CL1",
            config_id="C1",
            vehicle_type="Type1",
            customers=["A", "B"],
            total_demand={"Dry": 10, "Frozen": 5},
            centroid_latitude=4.5,
            centroid_longitude=-74.0,
            goods_in_config=["Dry", "Frozen"],
            route_time=3.5,
            method="kmeans",
            tsp_sequence=["Depot", "A", "B", "Depot"],
        ),
        Cluster(
            cluster_id="CL2",
            config_id="C2",
            vehicle_type="Type2",
            customers=[],
            total_demand={},
            centroid_latitude=0.0,
            centroid_longitude=0.0,
            goods_in_config=[],
            route_time=0.0,
        ),
    ]

    df = cluster_conversion.clusters_to_dataframe(clusters)

    assert set(df.columns) >= {
        "Cluster_ID",
        "Config_ID",
        "Vehicle_Type",
        "Customers",
        "Total_Demand",
        "Centroid_Latitude",
        "Centroid_Longitude",
        "Goods_In_Config",
        "Route_Time",
    }

    row = df.loc[df["Cluster_ID"] == "CL1"].iloc[0]
    assert row["TSP_Sequence"] == ["Depot", "A", "B", "Depot"]


def test_dataframes_roundtrip_back_to_clusters():
    df = pd.DataFrame(
        [
            {
                "Cluster_ID": "CL1",
                "Config_ID": "C1",
                "Vehicle_Type": "Type1",
                "Customers": ["A", "B"],
                "Total_Demand": {"Dry": 10, "Frozen": 5},
                "Centroid_Latitude": 4.5,
                "Centroid_Longitude": -74.0,
                "Goods_In_Config": ["Dry", "Frozen"],
                "Route_Time": 3.5,
                "Method": "kmeans",
                "TSP_Sequence": ["Depot", "A", "B", "Depot"],
            },
            {
                "Cluster_ID": "CL2",
                "Config_ID": "C2",
                "Vehicle_Type": "Type2",
                "Customers": ["C"],
                "Total_Demand": {"Dry": 3},
                "Centroid_Latitude": 4.7,
                "Centroid_Longitude": -74.2,
                "Goods_In_Config": ["Dry"],
                "Route_Time": 4.5,
                "Method": "hierarchical",
            },
        ]
    )

    clusters = cluster_conversion.dataframe_to_clusters(df)

    assert len(clusters) == 2
    first = clusters[0]
    assert first.cluster_id == "CL1"
    assert first.tsp_sequence == ["Depot", "A", "B", "Depot"]
    assert first.goods_in_config == ["Dry", "Frozen"]

    second = clusters[1]
    assert second.cluster_id == "CL2"
    assert second.customers == ["C"]


