"""Core merging functionality for combining clusters.

This module provides the generate_merge_phase_clusters function and its helpers
that are used both in pre-optimization cluster generation and post-optimization
improvement phases.
"""

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import VehicleConfiguration
from fleetmix.registry import ROUTE_TIME_ESTIMATOR_REGISTRY
from fleetmix.utils.logging import FleetmixLogger
from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    make_rt_context,
)

logger = FleetmixLogger.get_logger(__name__)

# Cache for merged cluster route times
_merged_route_time_cache: dict[tuple[str, ...], tuple[float, list | None]] = {}


# Helper functions for working with List[VehicleConfiguration]
def _find_config_by_id(
    configurations: list[VehicleConfiguration], config_id: str
) -> VehicleConfiguration:
    """Find configuration by ID from list."""
    for config in configurations:
        if str(config.config_id) == str(config_id):
            return config
    raise KeyError(f"Configuration {config_id} not found")


def _create_config_lookup(
    configurations: list[VehicleConfiguration],
) -> dict[str, VehicleConfiguration]:
    """Create a dictionary lookup for configurations."""
    return {str(config.config_id): config for config in configurations}


def _configs_to_dataframe(configurations: list[VehicleConfiguration]) -> pd.DataFrame:
    """Convert configurations to DataFrame when pandas operations are needed."""
    return pd.DataFrame([config.to_dict() for config in configurations])


def _get_merged_route_time(
    customers: pd.DataFrame, config: VehicleConfiguration, params: Parameters
) -> tuple[float, list | None]:
    """
    Estimate (and cache) the route time and sequence for a merged cluster of customers.
    Uses the vehicle configuration's timing parameters.
    """
    key: tuple[str, ...] = tuple(sorted(customers["Customer_ID"]))
    if key in _merged_route_time_cache:
        return _merged_route_time_cache[key]

    # Create RouteTimeContext using the factory
    rt_context = make_rt_context(config, params.depot, params.prune_tsp)

    # Use the new interface with RouteTimeContext
    estimator_class = ROUTE_TIME_ESTIMATOR_REGISTRY.get(
        params.clustering["route_time_estimation"]
    )
    if estimator_class is None:
        raise ValueError(
            f"Unknown route time estimation method: {params.clustering['route_time_estimation']}"
        )

    estimator = estimator_class()
    time, sequence = estimator.estimate_route_time(customers, rt_context)

    _merged_route_time_cache[key] = (time, sequence)
    return time, sequence


def generate_merge_phase_clusters(
    selected_clusters: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    customers_df: pd.DataFrame,
    params: Parameters,
) -> pd.DataFrame:
    """Generate merged clusters from selected small clusters.

    This function identifies small clusters that can be merged with nearby clusters
    to potentially reduce costs. It checks capacity, compartment compatibility, and
    route time constraints.

    Args:
        selected_clusters: DataFrame of currently selected clusters
        configurations: List of vehicle configurations
        customers_df: DataFrame of all customers
        params: Parameters object with merge configuration

    Returns:
        DataFrame of merged cluster candidates
    """
    new_clusters = []
    stats = {
        "attempted": 0,
        "valid": 0,
        "invalid_time": 0,
        "invalid_capacity": 0,
        "invalid_compatibility": 0,
    }

    # Create an indexed DataFrame for efficient configuration lookups
    configs_indexed = _configs_to_dataframe(configurations).set_index("Config_ID")
    # Index customers for fast lookup
    customers_indexed = customers_df.set_index("Customer_ID")

    # Start from selected_clusters (which already has goods columns) and add capacity
    cluster_meta = selected_clusters.copy()
    cluster_meta["Capacity"] = cluster_meta["Config_ID"].map(
        configs_indexed["Capacity"]
    )
    small_meta = cluster_meta[
        cluster_meta["Customers"].apply(len) <= params.small_cluster_size
    ]
    if small_meta.empty:
        return pd.DataFrame()
    target_meta = cluster_meta

    logger.debug(f"→ Found {len(small_meta)} small clusters")

    # Precompute numpy arrays for vectorized capacity & goods checks
    goods_arr = target_meta[params.goods].to_numpy()
    cap_arr = target_meta["Capacity"].to_numpy()
    ids = target_meta["Cluster_ID"].to_numpy()
    lat_arr = target_meta["Centroid_Latitude"].to_numpy()
    lon_arr = target_meta["Centroid_Longitude"].to_numpy()

    # Vectorized filtering loop
    for _, small in small_meta.iterrows():
        sd = np.array([small["Total_Demand"][g] for g in params.goods])
        peak = sd.max()
        total_small = sd.sum()
        needs = sd > 0
        if needs.any():
            goods_ok = (goods_arr[:, needs] == 1).all(axis=1)
        else:
            goods_ok = np.ones_like(cap_arr, dtype=bool)
        cap_ok = (cap_arr >= peak) & (cap_arr >= total_small)
        not_self = ids != small["Cluster_ID"]

        # Proximity-based filtering: compute distances and pick nearest candidates
        small_point = (small["Centroid_Latitude"], small["Centroid_Longitude"])
        target_points = np.column_stack((lat_arr, lon_arr))

        distances = haversine_vector(
            small_point, target_points, unit=Unit.KILOMETERS, comb=True
        )
        distances = distances.flatten()  # Ensure distances is a 1D array

        valid_mask = cap_ok & goods_ok & not_self & ~np.isnan(distances)
        valid_idxs = np.where(valid_mask)[0]
        if valid_idxs.size == 0:
            continue
        nearest_idxs = valid_idxs[
            np.argsort(distances[valid_idxs])[: params.nearest_merge_candidates]
        ]
        # Check capacity for total demand based on the memory
        total_small_sum = total_small

        for idx in nearest_idxs:
            target = target_meta.iloc[idx]

            # Get total demand for target cluster
            target_demand_total = sum(target["Total_Demand"][g] for g in params.goods)

            # Check if merging would exceed capacity (from memory)
            if total_small_sum + target_demand_total > target["Capacity"]:
                stats["invalid_capacity"] = stats.get("invalid_capacity", 0) + 1
                continue

            # ------------------------------------------------------------------
            # Avoid symmetric duplicates: process each unordered pair only once.
            # We enforce an arbitrary canonical ordering based on the string
            # representation of the cluster IDs.  If the current (small, target)
            # violates this order we simply skip it – the pair will (or has)
            # be(en) considered in the opposite direction.
            # ------------------------------------------------------------------
            small_id_str = str(small["Cluster_ID"])
            target_id_str = str(target["Cluster_ID"])
            if small_id_str > target_id_str:
                continue  # symmetric case already (or will be) handled

            # Quick lower-bound time prune before costly route-time estimation
            rt_target = target["Route_Time"]
            rt_small = small["Route_Time"]

            # Get the target configuration to access its service_time
            target_config = _find_config_by_id(configurations, target["Config_ID"])

            # Compute service time for the cluster not contributing the max route_time (avoid double count)
            if rt_small > rt_target:
                svc_time_other = calculate_total_service_time_hours(
                    len(target["Customers"]), target_config.service_time
                )
            else:
                svc_time_other = calculate_total_service_time_hours(
                    len(small["Customers"]), target_config.service_time
                )

            # Quick lower-bound time prune before costly route-time estimation (no proximity term)
            lb = max(rt_target, rt_small) + svc_time_other
            if lb > target_config.max_route_time:
                stats["invalid_time"] = stats.get("invalid_time", 0) + 1
                logger.debug(
                    f"Lower-bound prune: merge {small['Cluster_ID']} + {target['Cluster_ID']} lb={lb:.2f} > max={target_config.max_route_time:.2f}"
                )
                continue
            stats["attempted"] += 1
            is_valid, route_time, demands, tsp_sequence = validate_merged_cluster(
                small, target, target_config, customers_indexed, params
            )
            if not is_valid:
                # Assuming validate_merged_cluster now logs reasons for invalidity if needed
                # or updates stats for invalid_capacity, invalid_compatibility
                continue
            stats["valid"] += 1

            # Build merged cluster
            merged_customer_ids = target["Customers"] + small["Customers"]
            # Ensure merged_customers are fetched correctly for centroid calculation
            # It's crucial that customers_indexed contains all relevant customers.
            # If validate_merged_cluster already did this, we might optimize, but safety first.
            current_merged_customers_df = customers_indexed.loc[
                merged_customer_ids
            ].reset_index()

            centroid_lat = current_merged_customers_df["Latitude"].mean()
            centroid_lon = current_merged_customers_df["Longitude"].mean()
            # Build a canonical, order‐independent Cluster_ID
            canonical_id = (
                f"{small_id_str}_{target_id_str}"
                if small_id_str < target_id_str
                else f"{target_id_str}_{small_id_str}"
            )

            new_cluster = {
                "Cluster_ID": canonical_id,
                "Config_ID": target["Config_ID"],
                "Customers": merged_customer_ids,
                "Route_Time": route_time,
                "Total_Demand": demands,
                "Method": f"merged_{target['Method']}",
                "Centroid_Latitude": centroid_lat,
                "Centroid_Longitude": centroid_lon,
            }
            if tsp_sequence is not None:
                new_cluster["TSP_Sequence"] = tsp_sequence
            for good in params.goods:
                new_cluster[good] = target_config[good]
            new_clusters.append(new_cluster)

    # Log prune statistics before returning
    logger.debug(
        f"→ Merge prune stats: attempted={stats['attempted']}, "
        f"invalid_time={stats['invalid_time']}, "
        f"invalid_capacity={stats['invalid_capacity']}, "
        f"invalid_compatibility={stats['invalid_compatibility']}, "
        f"valid={stats['valid']}"
    )
    if not new_clusters:
        return pd.DataFrame()

    # Create barebones DataFrame with the minimal required columns
    # The improve_solution function will handle adding any missing columns
    minimal_columns = [
        "Cluster_ID",
        "Config_ID",
        "Customers",
        "Route_Time",
        "Total_Demand",
        "Method",
        "Centroid_Latitude",
        "Centroid_Longitude",
        "TSP_Sequence",
    ] + list(params.goods)

    # Build and dedupe merged clusters
    df = pd.DataFrame(new_clusters, columns=minimal_columns)
    return df.drop_duplicates("Cluster_ID")


def validate_merged_cluster(
    cluster1: pd.Series,
    cluster2: pd.Series,
    config: VehicleConfiguration,
    customers_df: pd.DataFrame,
    params: Parameters,
) -> tuple[bool, float, dict, list | None]:
    """Validate if two clusters can be merged using the given vehicle configuration.

    Args:
        cluster1: First cluster to merge
        cluster2: Second cluster to merge
        config: Vehicle configuration to use for the merged cluster
        customers_df: DataFrame of all customers
        params: Parameters object

    Returns:
        Tuple of (is_valid, route_time, demands_dict, tsp_sequence)
    """
    # Index customers for fast lookup
    if customers_df.index.name != "Customer_ID":
        customers_indexed = customers_df.set_index("Customer_ID", drop=False)
    else:
        customers_indexed = customers_df
    # Check compartment compatibility
    merged_goods = {}
    for g in params.goods:
        # Handle case where Total_Demand might be a dict or series
        demand1 = (
            cluster1["Total_Demand"][g]
            if isinstance(cluster1["Total_Demand"], (dict, pd.Series))
            else cluster1[g]
        )
        demand2 = (
            cluster2["Total_Demand"][g]
            if isinstance(cluster2["Total_Demand"], (dict, pd.Series))
            else cluster2[g]
        )
        merged_goods[g] = demand1 + demand2

    # Validate capacity – per-good and total demand
    if any(demand > config.capacity for demand in merged_goods.values()):
        return False, 0, {}, None

    total_demand = sum(merged_goods.values())
    if total_demand > config.capacity:
        return False, 0, {}, None

    # Get all customers from both clusters
    cluster1_customers = (
        cluster1["Customers"]
        if isinstance(cluster1["Customers"], list)
        else [cluster1["Customers"]]
    )
    cluster2_customers = (
        cluster2["Customers"]
        if isinstance(cluster2["Customers"], list)
        else [cluster2["Customers"]]
    )

    merged_customers_ids = cluster1_customers + cluster2_customers
    # Validate that all customer IDs are present in customers_indexed
    # Check if customers_indexed has 'Customer_ID' as its index
    if customers_indexed.index.name != "Customer_ID":
        # This case should ideally not happen if indexing is consistent
        logger.error(
            "customers_indexed is not indexed by 'Customer_ID' in validate_merged_cluster."
        )
        # Fallback or raise error, for now, assume it's an issue and return invalid
        return False, 0, {}, None

    missing_ids = [
        cid for cid in merged_customers_ids if cid not in customers_indexed.index
    ]
    if missing_ids:
        logger.warning(
            f"Missing customer IDs {missing_ids} during merge validation for potential merge of clusters involving {cluster1.get('Cluster_ID', 'Unknown')} and {cluster2.get('Cluster_ID', 'Unknown')}."
        )
        return False, 0, {}, None

    merged_customers = customers_indexed.loc[merged_customers_ids].reset_index()

    # Validate customer locations
    if (
        merged_customers["Latitude"].isna().any()
        or merged_customers["Longitude"].isna().any()
    ):
        return False, 0, {}, None

    # Estimate (and cache) new route time using the general estimator
    new_route_time, new_sequence = _get_merged_route_time(
        merged_customers, config, params
    )

    if new_route_time > config.max_route_time:
        return False, 0, {}, None

    return True, new_route_time, merged_goods, new_sequence
