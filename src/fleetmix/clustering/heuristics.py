"""
heuristics.py

This module implements the clustering heuristics algorithms used in the cluster generation process.
It contains the lower-level implementation details of the clustering algorithms, constraint checking,
and recursive splitting logic.
"""

from typing import Any

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import (
    Cluster,
    ClusteringContext,
    Customer,
    CustomerBase,
    VehicleConfiguration,
)
from fleetmix.registry import (
    CLUSTERER_REGISTRY,
    ROUTE_TIME_ESTIMATOR_REGISTRY,
    register_clusterer,
)
from fleetmix.utils.logging import FleetmixLogger
from fleetmix.utils.route_time import make_rt_context

logger = FleetmixLogger.get_logger(__name__)

# Product weights for demand profile calculations - equal weighting for all product types
PRODUCT_WEIGHTS = {
    "Frozen": 1.0 / 3.0,  # Equal priority (1/3)
    "Chilled": 1.0 / 3.0,  # Equal priority (1/3)
    "Dry": 1.0 / 3.0,  # Equal priority (1/3)
}


@register_clusterer("minibatch_kmeans")
class MiniBatchKMeansClusterer:
    """MiniBatch KMeans clustering algorithm."""

    def fit(
        self, customers: pd.DataFrame, *, context: ClusteringContext, n_clusters: int
    ) -> list[int]:
        """Cluster customers using MiniBatch KMeans."""
        data = compute_cluster_metric_input(customers, context, "minibatch_kmeans")
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        # Convert numpy array to list of ints
        return [int(label) for label in labels]


@register_clusterer("kmedoids")
class KMedoidsClusterer:
    """K-Medoids clustering algorithm."""

    def fit(
        self, customers: pd.DataFrame, *, context: ClusteringContext, n_clusters: int
    ) -> list[int]:
        """Cluster customers using K-Medoids."""
        data = compute_cluster_metric_input(customers, context, "kmedoids")
        model = KMedoids(
            n_clusters=n_clusters,
            metric="euclidean",
            method="fasterpam",
            init="build",
            max_iter=300,
            random_state=42,
        )
        labels = model.fit_predict(data)
        # Convert numpy array to list of ints
        return [int(label) for label in labels]


@register_clusterer("agglomerative")
class AgglomerativeClusterer:
    """Agglomerative clustering algorithm."""

    def fit(
        self, customers: pd.DataFrame, *, context: ClusteringContext, n_clusters: int
    ) -> list[int]:
        """Cluster customers using Agglomerative clustering."""
        # Agglomerative clustering needs precomputed distance matrix
        data = compute_cluster_metric_input(customers, context, "agglomerative")
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        labels = model.fit_predict(data)
        # Convert numpy array to list of ints
        return [int(label) for label in labels]


@register_clusterer("gaussian_mixture")
class GaussianMixtureClusterer:
    """Gaussian Mixture Model clustering algorithm."""

    def fit(
        self, customers: pd.DataFrame, *, context: ClusteringContext, n_clusters: int
    ) -> list[int]:
        """Cluster customers using Gaussian Mixture Model."""
        data = compute_cluster_metric_input(customers, context, "gaussian_mixture")
        model = GaussianMixture(
            n_components=n_clusters, random_state=42, covariance_type="full"
        )
        labels = model.fit_predict(data)
        # Convert numpy array to list of ints
        return [int(label) for label in labels]


def compute_cluster_metric_input(
    customers: pd.DataFrame, context: ClusteringContext, method: str
) -> np.ndarray:
    """Get appropriate input for clustering algorithm."""
    # Methods that need precomputed distance matrix
    needs_precomputed = method.startswith("agglomerative")

    if needs_precomputed:
        logger.debug(
            f"Using precomputed distance matrix for method {method} with geo_weight={context.geo_weight}, demand_weight={context.demand_weight}"
        )
        return compute_composite_distance(
            customers, context.goods, context.geo_weight, context.demand_weight
        )
    else:
        logger.debug(f"Using feature-based input for method {method}")
        # Ensure data is in the right format - contiguous float64 array
        data = customers[["Latitude", "Longitude"]].values
        return np.ascontiguousarray(data, dtype=np.float64)


def compute_composite_distance(
    customers: pd.DataFrame, goods: list[str], geo_weight: float, demand_weight: float
) -> np.ndarray:
    """Compute composite distance matrix combining geographical and demand distances."""
    # Compute geographical distance
    coords = customers[["Latitude", "Longitude"]].values
    geo_dist = pairwise_distances(coords, metric="euclidean")

    # Compute demand profiles
    demands = customers[[f"{g}_Demand" for g in goods]].fillna(0).values
    demand_profiles = np.zeros_like(demands, dtype=float)

    # Convert to proportions
    total_demands = demands.sum(axis=1)
    nonzero_mask = total_demands > 0
    for i in range(len(goods)):
        demand_profiles[nonzero_mask, i] = (
            demands[nonzero_mask, i] / total_demands[nonzero_mask]
        )

    # Apply temperature sensitivity weights
    for i, good in enumerate(goods):
        demand_profiles[:, i] *= PRODUCT_WEIGHTS.get(good, 1.0)

    # Compute demand similarity using cosine distance
    demand_dist = pairwise_distances(demand_profiles, metric="cosine")
    demand_dist = np.nan_to_num(demand_dist, nan=1.0)

    # If dealing with pseudo-customers, force demand distance to 0 for same-origin customers
    if "Origin_ID" in customers.columns:
        origin_ids = customers["Origin_ID"].values
        unique_origins = pd.unique(origin_ids[pd.notna(origin_ids)])

        for origin_id in unique_origins:
            indices = np.where(origin_ids == origin_id)[0]
            if len(indices) > 1:
                # Create all pairs of indices for this origin_id
                idx_pairs = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)
                # Set demand distance to 0 for these pairs
                demand_dist[idx_pairs[:, 0], idx_pairs[:, 1]] = 0.0

    # Normalize distances
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()

    # Combine distances with weights
    composite_distance: np.ndarray = (geo_weight * geo_dist) + (
        demand_weight * demand_dist
    )

    return composite_distance


def get_cached_demand(
    customers: list[CustomerBase],
    goods: list[str],
    demand_cache: dict[Any, dict[str, float]],
) -> dict[str, float]:
    """Get demand from cache or compute and cache it."""
    # Use sorted tuple of customer IDs as key (immutable and hashable)
    key = tuple(sorted(customer.customer_id for customer in customers))

    # Check if in cache
    cached_result = demand_cache.get(key)
    if cached_result is not None:
        return cached_result

    # Not in cache, compute it. This handles regular, pseudo, and mixed customer lists.
    demand_dict = {g: 0.0 for g in goods}
    origin_demands: dict[str, dict[str, float]] = {}  # For pseudo-customers

    for customer in customers:
        if customer.is_pseudo_customer():
            origin_id = customer.get_origin_id()
            if origin_id not in origin_demands:
                origin_demands[origin_id] = {g: 0.0 for g in goods}

            # Only add demand for goods in this pseudo-customer's subset
            goods_subset = customer.get_goods_subset()
            for good in goods_subset:
                good_lower = good.lower()
                # Find the matching good in the customer's demands (case-insensitive)
                demand_key = next(
                    (k for k in customer.demands if k.lower() == good_lower), None
                )
                if demand_key:
                    # Find the matching good in the goods list (case-insensitive)
                    goods_key = next(
                        (g for g in goods if g.lower() == good_lower), None
                    )
                    if goods_key:
                        origin_demands[origin_id][goods_key] = max(
                            origin_demands[origin_id][goods_key],
                            customer.demands[demand_key],
                        )
        else:
            # For regular customers, simple sum
            for good in goods:
                good_lower = good.lower()
                demand_value = next(
                    (v for k, v in customer.demands.items() if k.lower() == good_lower),
                    0.0,
                )
                demand_dict[good] += demand_value

    # Add pseudo-customer demands to the total
    for single_origin_demands in origin_demands.values():
        for good, value in single_origin_demands.items():
            demand_dict[good] += value

    # Store in cache
    demand_cache[key] = demand_dict
    return demand_dict


def get_cached_route_time(
    customers: list[CustomerBase],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    route_time_cache: dict[Any, tuple[float, list[str]]],
    main_params: Parameters,
) -> tuple[float, list[str]]:
    """Get route time and sequence (if TSP) from cache or compute and cache it."""
    key = tuple(sorted(customer.customer_id for customer in customers))
    cached_result = route_time_cache.get(key)
    if cached_result is not None:
        return cached_result

    # Convert to DataFrame for route time estimation (temporary until we refactor route time estimators)
    customers_df = Customer.to_dataframe(customers)

    # Create RouteTimeContext using the factory
    rt_context = make_rt_context(
        config, clustering_context.depot, main_params.prune_tsp
    )

    # Use the new interface with RouteTimeContext
    estimator_class = ROUTE_TIME_ESTIMATOR_REGISTRY.get(
        clustering_context.route_time_estimation
    )
    if estimator_class is None:
        raise ValueError(
            f"Unknown route time estimation method: {clustering_context.route_time_estimation}"
        )

    estimator = estimator_class()
    route_time, route_sequence = estimator.estimate_route_time(customers_df, rt_context)

    result = (route_time, route_sequence)
    route_time_cache[key] = result
    return result


def get_feasible_customers_subset(
    customers: list[CustomerBase], feasible_customers: dict, config_id: int
) -> list[CustomerBase]:
    """Extract feasible customers for a given configuration."""
    return [
        customer
        for customer in customers
        if customer.customer_id in feasible_customers
        and config_id in feasible_customers[customer.customer_id]
    ]


def create_initial_clusters(
    customers_subset: list[CustomerBase],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    main_params: Parameters,
    method_name: str = "minibatch_kmeans",
) -> list[list[CustomerBase]]:
    """Create initial clusters for the given customer subset."""
    if len(customers_subset) <= 2:
        return create_small_dataset_clusters(customers_subset)
    else:
        return create_normal_dataset_clusters(
            customers_subset, config, clustering_context, main_params, method_name
        )


def create_small_dataset_clusters(
    customers_subset: list[CustomerBase],
) -> list[list[CustomerBase]]:
    """Create clusters for small datasets (≤2 customers)."""
    # Put all customers in a single cluster
    return [customers_subset]


def create_normal_dataset_clusters(
    customers_subset: list[CustomerBase],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    main_params: Parameters,
    method_name: str,
) -> list[list[CustomerBase]]:
    """Create clusters for normal-sized datasets."""
    # Convert to DataFrame temporarily for clustering algorithm
    customers_df = Customer.to_dataframe(customers_subset)

    # Determine number of clusters
    num_clusters = estimate_num_initial_clusters(
        customers_df, config, clustering_context, main_params
    )

    # Ensure the number of clusters doesn't exceed the number of customers
    num_clusters = min(num_clusters, len(customers_subset))

    # Get the clusterer from registry
    clusterer_class = CLUSTERER_REGISTRY.get(method_name)
    if clusterer_class is None:
        logger.error(f"❌ Unknown clustering method: {method_name}")
        raise ValueError(f"Unknown clustering method: {method_name}")

    # Create instance and cluster
    clusterer = clusterer_class()
    labels = clusterer.fit(
        customers_df, context=clustering_context, n_clusters=num_clusters
    )

    # Group customers by cluster label
    clusters = []
    for cluster_label in range(num_clusters):
        cluster_customers = [
            customers_subset[i]
            for i, label in enumerate(labels)
            if label == cluster_label
        ]
        if cluster_customers:  # Only add non-empty clusters
            clusters.append(cluster_customers)

    return clusters


def generate_cluster_id_base(config_id: int) -> int:
    """Generate a base cluster ID from the configuration ID."""
    return int(str(config_id) + "000")


def check_constraints(
    cluster_customers: list[CustomerBase],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    demand_cache: dict[Any, dict[str, float]],
    route_time_cache: dict[Any, tuple[float, list[str]]],
    main_params: Parameters,
) -> tuple[bool, bool]:
    """
    Check if cluster violates capacity or time constraints.

    Returns:
        tuple: (capacity_violated, time_violated)
    """
    # Get demand from cache
    demand_dict = get_cached_demand(
        cluster_customers, clustering_context.goods, demand_cache
    )
    cluster_demand = sum(demand_dict.values())

    # Get route time from cache (ignore sequence for constraint check)
    route_time, _ = get_cached_route_time(
        cluster_customers, config, clustering_context, route_time_cache, main_params
    )

    capacity_violated = cluster_demand > config.capacity
    time_violated = route_time > config.max_route_time

    return capacity_violated, time_violated


def should_split_cluster(
    cluster_customers: list[CustomerBase],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    depth: int,
    demand_cache: dict[Any, dict[str, float]],
    route_time_cache: dict[Any, tuple[float, list[str]]],
    main_params: Parameters,
) -> bool:
    """Determine if a cluster should be split based on constraints."""
    capacity_violated, time_violated = check_constraints(
        cluster_customers,
        config,
        clustering_context,
        demand_cache,
        route_time_cache,
        main_params,
    )
    is_singleton_cluster = len(cluster_customers) <= 1

    # Log warning for single-customer constraints
    if (capacity_violated or time_violated) and is_singleton_cluster:
        logger.debug(
            f"⚠️ Can't split further (singleton cluster) but constraints still violated: "
            f"capacity={capacity_violated}, time={time_violated}"
        )

    # Return True if we need to split (constraints violated and we can split)
    return (capacity_violated or time_violated) and not is_singleton_cluster


def split_cluster(
    cluster_customers: list[CustomerBase],
    clustering_context: ClusteringContext,
    method_name: str,
) -> list[list[CustomerBase]]:
    """Split an oversized cluster into smaller ones."""
    # Convert to DataFrame temporarily for clustering algorithm
    customers_df = Customer.to_dataframe(cluster_customers)

    # Get the clusterer from registry
    clusterer_class = CLUSTERER_REGISTRY.get(method_name)
    if clusterer_class is None:
        logger.error(f"❌ Unknown clustering method: {method_name}")
        raise ValueError(f"Unknown clustering method: {method_name}")

    # Create instance and split into 2 clusters
    clusterer = clusterer_class()
    sub_labels_list = clusterer.fit(
        customers_df, context=clustering_context, n_clusters=2
    )

    # Convert list to numpy array for indexing
    sub_labels = np.array(sub_labels_list)

    # Create sub-clusters
    sub_clusters = []
    sub_cluster_sizes = []
    for label in [0, 1]:
        mask = sub_labels == label
        sub_cluster = [
            cluster_customers[i]
            for i, is_in_cluster in enumerate(mask)
            if is_in_cluster
        ]
        if sub_cluster:  # Only add non-empty clusters
            sub_clusters.append(sub_cluster)
            sub_cluster_sizes.append(len(sub_cluster))

    logger.debug(
        f"Split cluster of size {len(cluster_customers)} into {len(sub_clusters)} "
        f"sub-clusters of sizes {sub_cluster_sizes}"
    )

    return sub_clusters


def create_cluster(
    cluster_customers: list[CustomerBase],
    config: VehicleConfiguration,
    cluster_id: int,
    clustering_context: ClusteringContext,
    demand_cache: dict[Any, dict[str, float]],
    route_time_cache: dict[Any, tuple[float, list[str]]],
    main_params: Parameters,
    method_name: str,
) -> Cluster:
    """Create a Cluster object from customer data."""
    # Get demand from cache
    total_demand = get_cached_demand(
        cluster_customers, clustering_context.goods, demand_cache
    )

    # Get route time and sequence from cache
    route_time, tsp_sequence = get_cached_route_time(
        cluster_customers, config, clustering_context, route_time_cache, main_params
    )

    # Calculate centroid
    if cluster_customers:
        centroid_latitude = sum(
            customer.location[0] for customer in cluster_customers
        ) / len(cluster_customers)
        centroid_longitude = sum(
            customer.location[1] for customer in cluster_customers
        ) / len(cluster_customers)
    else:
        centroid_latitude = 0.0
        centroid_longitude = 0.0

    cluster = Cluster(
        cluster_id=cluster_id,
        config_id=config.config_id,
        customers=[customer.customer_id for customer in cluster_customers],
        total_demand=total_demand,
        centroid_latitude=centroid_latitude,
        centroid_longitude=centroid_longitude,
        goods_in_config=[g for g in clustering_context.goods if config.compartments[g]],
        route_time=route_time,
        method=method_name,
        tsp_sequence=tsp_sequence,
    )
    return cluster


def process_clusters_recursively(
    initial_clusters: list[list[CustomerBase]],
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    demand_cache: dict[Any, dict[str, float]],
    route_time_cache: dict[Any, tuple[float, list[str]]],
    main_params: Parameters | None = None,
    method_name: str = "minibatch_kmeans",
) -> list[Cluster]:
    """Process clusters recursively to ensure constraints are satisfied."""
    if main_params is None:
        raise ValueError("main_params is required for cluster processing")

    config_id = config.config_id
    cluster_id_base = generate_cluster_id_base(config_id)
    current_cluster_id = 0
    clusters = []

    # Process clusters until all constraints are satisfied
    clusters_to_check = [
        (cluster_customers, 0) for cluster_customers in initial_clusters
    ]

    logger.info(
        f"Starting recursive processing for config {config_id} with {len(clusters_to_check)} initial clusters"
    )

    split_count = 0
    skipped_count = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        max_depth_reached = depth >= clustering_context.max_depth

        # Check if max depth reached
        if max_depth_reached:
            # Check if constraints violated
            capacity_violated, time_violated = check_constraints(
                cluster_customers,
                config,
                clustering_context,
                demand_cache,
                route_time_cache,
                main_params,
            )

            if capacity_violated or time_violated:
                logger.debug(
                    f"⚠️ Max depth {clustering_context.max_depth} reached but constraints still violated: "
                    f"capacity={capacity_violated}, time={time_violated}, "
                    f"method={method_name}, config_id={config.config_id}"
                )
                skipped_count += 1
                continue  # Skip this cluster

        # Not at max depth, check if we should split
        if not max_depth_reached and should_split_cluster(
            cluster_customers,
            config,
            clustering_context,
            depth,
            demand_cache,
            route_time_cache,
            main_params,
        ):
            split_count += 1
            logger.debug(
                f"Splitting cluster for config {config_id} (size {len(cluster_customers)}) at depth {depth}/{clustering_context.max_depth}"
            )
            # Split oversized clusters
            for sub_cluster in split_cluster(
                cluster_customers, clustering_context, method_name
            ):
                clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster (either constraints satisfied or could not be split further)
            current_cluster_id += 1
            cluster = create_cluster(
                cluster_customers,
                config,
                cluster_id_base + current_cluster_id,
                clustering_context,
                demand_cache,
                route_time_cache,
                main_params,
                method_name,
            )
            clusters.append(cluster)

    if skipped_count > 0:
        logger.debug(
            f"⚠️ Skipped {skipped_count} clusters that exceeded capacity at max depth for config {config_id}."
        )

    logger.info(
        f"Completed recursive processing for config {config_id}: {len(clusters)} final clusters, "
        f"{split_count} splits performed"
    )

    return clusters


def estimate_num_initial_clusters(
    customers: pd.DataFrame,
    config: VehicleConfiguration,
    clustering_context: ClusteringContext,
    main_params: Parameters | None = None,
) -> int:
    """
    Estimate the number of initial clusters needed based on capacity and time constraints.
    """
    if customers.empty:
        return 0

    # Default prune_tsp flag if main_params not provided
    prune_tsp_val = main_params.prune_tsp if main_params is not None else False

    # Convert to CustomerBase objects and determine if they are pseudo-customers
    customers_list = Customer.from_dataframe(customers)
    is_pseudo = bool(customers_list and customers_list[0].is_pseudo_customer())

    # Calculate total demand. For pseudo-customers, this is aggregated by origin.
    demand_cache: dict[Any, dict[str, float]] = {}
    demand_dict = get_cached_demand(
        customers_list, clustering_context.goods, demand_cache
    )

    total_demand = 0.0
    for good in clustering_context.goods:
        if config.compartments.get(good):  # Only consider goods this vehicle can carry
            total_demand += demand_dict.get(good, 0.0)

    # Estimate clusters needed based on capacity
    if total_demand == 0:
        clusters_by_capacity = 1.0  # At least one cluster needed even with zero demand
    else:
        clusters_by_capacity = np.ceil(total_demand / config.capacity)

    # For pseudo-customers, count unique origins for stop count and sample from unique locations
    if is_pseudo:
        num_stops = customers["Origin_ID"].nunique()
        # Sample from unique origins to get a representative cluster for time estimation
        sample_customers_df = customers.drop_duplicates(subset=["Origin_ID"])
    else:
        num_stops = len(customers)
        sample_customers_df = customers

    # Estimate time for an average route
    avg_customers_per_cluster = (
        num_stops / clusters_by_capacity if clusters_by_capacity > 0 else num_stops
    )

    # Ensure sample size doesn't exceed population size and is at least 1 if possible
    if (
        np.isinf(avg_customers_per_cluster)
        or np.isnan(avg_customers_per_cluster)
        or avg_customers_per_cluster < 1
    ):
        sample_size = num_stops
    else:
        sample_size = max(1, min(int(avg_customers_per_cluster), num_stops))

    if sample_size > 0:
        avg_cluster = sample_customers_df.sample(n=sample_size)
    else:
        avg_cluster = pd.DataFrame(columns=customers.columns)

    # Create RouteTimeContext using the factory
    rt_context = make_rt_context(config, clustering_context.depot, prune_tsp_val)

    # Use the new interface with RouteTimeContext
    estimator_class = ROUTE_TIME_ESTIMATOR_REGISTRY.get(
        clustering_context.route_time_estimation
    )
    if estimator_class is None:
        raise ValueError(
            f"Unknown route time estimation method: {clustering_context.route_time_estimation}"
        )

    estimator = estimator_class()
    avg_route_time, _ = estimator.estimate_route_time(avg_cluster, rt_context)

    # Estimate clusters needed based on time
    if config.max_route_time > 0 and avg_customers_per_cluster > 0:
        clusters_by_time = np.ceil(
            avg_route_time
            * num_stops
            / (config.max_route_time * avg_customers_per_cluster)
        )
    else:
        clusters_by_time = 1.0

    # Take the maximum of the two estimates
    num_clusters = int(max(clusters_by_capacity, clusters_by_time, 1))

    logger.debug(
        f"Estimated clusters: {num_clusters} "
        f"(capacity: {clusters_by_capacity}, time: {clusters_by_time})"
    )

    return num_clusters
