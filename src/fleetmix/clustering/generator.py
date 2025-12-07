"""
generator.py

Main module for generating capacity- and time-feasible customer clusters for the fleet design
optimization process. This is the main entry point for the cluster-first phase of the
cluster-first, fleet-design second heuristic.
"""

import itertools
import os
from dataclasses import replace
from multiprocessing import Manager
from typing import Dict, List

from joblib import Parallel, delayed

from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import (
    CapacitatedClusteringContext,
    Cluster,
    Customer,
    CustomerBase,
    VehicleConfiguration,
)
from fleetmix.merging.core import generate_merge_phase_clusters
from fleetmix.utils.common import to_cfg_key
from fleetmix.utils.logging import FleetmixLogger

from .heuristics import (
    create_initial_clusters,
    get_feasible_customers_subset,
    process_clusters_recursively,
)

logger = FleetmixLogger.get_logger(__name__)


class Symbols:
    """Unicode symbols for logging."""

    CHECKMARK = "✓"
    CROSS = "✗"


def generate_feasible_clusters(
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    params: FleetmixParams,
) -> list[Cluster]:
    """
    Generate clusters for each vehicle configuration in parallel.

    Args:
        customers: List of CustomerBase objects containing customer data
        configurations: List of vehicle configurations
        params: Parameters object containing vehicle configuration parameters

    Returns:
        List of Cluster objects containing all generated clusters
    """

    FleetmixLogger.detail("--- Starting Cluster Generation Process ---")
    if not customers or not configurations:
        logger.warning(
            "Input customers or configurations are empty. Returning empty list."
        )
        return []

    with Manager() as manager:
        shared_demand_cache = manager.dict()
        shared_route_time_cache = manager.dict()

        logger.debug(
            "Initializing shared caches for demand and route time calculations"
        )

        # 1. Generate feasibility mapping
        FleetmixLogger.detail("Generating feasibility mapping...")
        feasible_customers = _generate_feasibility_mapping(
            customers, configurations, params.problem.goods
        )
        if not feasible_customers:
            logger.warning(
                "No customers are feasible for any configuration. Returning empty list."
            )
            return []
        FleetmixLogger.detail(
            f"Feasibility mapping generated for {len(feasible_customers)} customers."
        )

        # 2. Generate list of (ClusteringContext, method_name) tuples for all runs
        context_and_methods = _get_clustering_context_list(params)

        # 3. Precompute distance/duration matrices if TSP route estimation is used
        tsp_needed = any(
            clustering_context.route_time_estimation == "TSP"
            for clustering_context, _ in context_and_methods
        )
        if tsp_needed:
            FleetmixLogger.warning(
                "Using TSP for route time estimation (will be slower than BHH). "
                "If using 'combine' method, consider switching to a single method (e.g. 'minibatch_kmeans') to speed up TSP computations. "
                "Alternatively, set route_time_estimation: 'BHH' in config for faster results."
            )
            logger.debug(
                "TSP route estimation detected. Building distance/duration matrices per vehicle configuration..."
            )
            # Build matrices for each unique avg_speed value across configurations
            from fleetmix.utils.route_time import build_distance_duration_matrices

            unique_speeds = set(config.avg_speed for config in configurations)
            for speed in unique_speeds:
                depot_dict = {
                    "latitude": params.problem.depot.latitude,
                    "longitude": params.problem.depot.longitude,
                }
                # Convert customers to DataFrame for matrix building (temporary)
                customers_df = Customer.to_dataframe(customers)
                build_distance_duration_matrices(customers_df, depot_dict, speed)
            logger.debug(f"Built matrices for speeds: {sorted(unique_speeds)} km/h")
        else:
            logger.debug(
                "TSP route estimation not used. Skipping matrix precomputation."
            )

        cluster_id_generator = itertools.count()

        # 4. Process configurations in parallel for each context configuration
        all_clusters = []
        for clustering_context, method_name in context_and_methods:
            logger.debug(
                f"Running configuration: {method_name} (GeoW: {clustering_context.geo_weight:.2f}, DemW: {clustering_context.demand_weight:.2f})"
            )

            # Determine level of parallelism: obey FLEETMIX_N_JOBS env var if set
            n_jobs_env = os.getenv("FLEETMIX_N_JOBS")
            try:
                n_jobs = int(n_jobs_env) if n_jobs_env is not None else -1
            except ValueError:
                # Fallback to default behaviour if parsing fails
                n_jobs = -1

            # Run clustering for all configurations using these context in parallel, process-based
            clusters_by_config = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_configuration)(
                    config,
                    customers,
                    feasible_customers,
                    clustering_context,
                    shared_demand_cache,
                    shared_route_time_cache,
                    params,
                    method_name,
                )
                for config in configurations
            )

            # Flatten the list of lists returned by Parallel and assign IDs
            for config_clusters in clusters_by_config:
                for cluster in config_clusters:
                    # Assign unique Cluster_ID
                    cluster.cluster_id = next(cluster_id_generator)
                    all_clusters.append(cluster)
            logger.debug(
                f"Configuration {method_name} completed: {len([c for config_clusters in clusters_by_config for c in config_clusters])} raw clusters"
            )

        logger.debug(
            f"Cache statistics: {len(shared_demand_cache)} demand entries, {len(shared_route_time_cache)} route time entries"
        )

    if not all_clusters:
        logger.warning("No clusters were generated by any configuration.")
        return []

    # Remove duplicate clusters based on customer sets
    FleetmixLogger.detail(
        f"Combining and deduplicating {len(all_clusters)} raw clusters from all configurations..."
    )
    unique_clusters = _deduplicate_clusters(all_clusters)

    # ------------------------------------------------------------------
    # Generate additional candidate clusters by merging neighbouring base
    # clusters ahead of the MILP optimisation. The same routine that is
    # normally applied after the MILP is reused here so that the solver
    # can already consider larger cluster combinations that may reduce
    # the required fleet size.
    # ------------------------------------------------------------------

    customers_df = Customer.to_dataframe(customers)
    base_df = Cluster.to_dataframe(unique_clusters)

    # Ensure goods indicator columns exist to satisfy downstream logic
    config_lookup = {str(cfg.config_id): cfg for cfg in configurations}
    for good in params.problem.goods:
        base_df[good] = base_df["Config_ID"].map(
            lambda x: config_lookup[str(x)][good] if str(x) in config_lookup else 0
        )
    if "Capacity" not in base_df.columns:
        base_df["Capacity"] = base_df["Config_ID"].map(
            lambda x: config_lookup[str(x)].capacity if str(x) in config_lookup else 0
        )

    # Generate additional clusters through vehicle-type-aware merging
    # The merging function now handles vehicle type checking internally
    merged_df = generate_merge_phase_clusters(
        selected_clusters=base_df,
        configurations=configurations,
        customers_df=customers_df,
        params=params,
        small_cluster_size=params.algorithm.pre_small_cluster_size,
        nearest_merge_candidates=params.algorithm.pre_nearest_merge_candidates,
    )

    if not merged_df.empty:
        next_id = max(c.cluster_id for c in unique_clusters) + 1
        merged_clusters = Cluster.from_dataframe(merged_df)
        for cluster in merged_clusters:
            cluster.cluster_id = next_id
            next_id += 1
            unique_clusters.append(cluster)

        FleetmixLogger.detail(
            f"Added {len(merged_clusters)} merged neighbour clusters (pre-MILP)"
        )

    # Final deduplication
    unique_clusters = _deduplicate_clusters(unique_clusters)

    # Validate cluster coverage
    validate_cluster_coverage(unique_clusters, customers)

    FleetmixLogger.detail("--- Cluster Generation Complete ---")
    FleetmixLogger.detail(
        f"Generated {len(unique_clusters)} unique clusters across all configurations"
    )

    return unique_clusters


def process_configuration(
    config: VehicleConfiguration,
    customers: list[CustomerBase],
    feasible_customers: dict,
    context: CapacitatedClusteringContext,
    demand_cache: dict | None = None,
    route_time_cache: dict | None = None,
    main_params: FleetmixParams | None = None,
    method_name: str = "minibatch_kmeans",
) -> list[Cluster]:
    """Process a single vehicle configuration to generate feasible clusters."""
    if main_params is None:
        raise ValueError("main_params is required for configuration processing")

    # Provide default empty dictionaries if caches are None
    demand_cache = demand_cache or {}
    route_time_cache = route_time_cache or {}

    # 1. Get customers that can be served by the configuration
    customers_subset = get_feasible_customers_subset(
        customers, feasible_customers, config.config_id
    )
    if not customers_subset:
        return []

    # 2. Create initial clusters (one large cluster for the subset)
    initial_clusters = create_initial_clusters(
        customers_subset, config, context, main_params, method_name
    )

    # 3. Process clusters recursively until constraints are satisfied
    return process_clusters_recursively(
        initial_clusters,
        config,
        context,
        demand_cache,
        route_time_cache,
        main_params,
        method_name,
    )


def validate_cluster_coverage(
    clusters: list[Cluster], customers: list[CustomerBase]
) -> None:
    """Validate that all customers are covered by at least one cluster."""
    customer_coverage = dict.fromkeys(
        [customer.customer_id for customer in customers], False
    )

    for cluster in clusters:
        for customer_id in cluster.customers:
            customer_coverage[customer_id] = True

    uncovered = [cid for cid, covered in customer_coverage.items() if not covered]

    if uncovered:
        logger.warning(
            f"Found {len(uncovered)} customers not covered by any cluster: {uncovered[:5]}..."
        )
    else:
        FleetmixLogger.detail(
            f"✓ All {len(customer_coverage)} customers are covered by at least one cluster."
        )


def _generate_feasibility_mapping(
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    goods: list[str],
) -> dict:
    """Generate mapping of feasible configurations for each customer."""
    feasible_customers = {}

    for customer in customers:
        customer_id = customer.customer_id
        feasible_configs = []

        for config in configurations:
            if _is_customer_feasible(customer, config, goods):
                feasible_configs.append(to_cfg_key(config.config_id))

        if feasible_configs:
            feasible_customers[customer_id] = feasible_configs

    logger.debug(
        f"Feasibility mapping: {len(feasible_customers)} customers have feasible configs"
    )

    return feasible_customers


def _is_customer_feasible(
    customer: CustomerBase, config: VehicleConfiguration, goods: list[str]
) -> bool:
    """Check if a customer's demands can be served by a configuration."""
    for good in goods:
        if customer.has_demand_for(good) and not config.compartments[good]:
            return False
        if customer.demands.get(good, 0.0) > config.capacity:
            return False
    return True


def _deduplicate_clusters(clusters: list[Cluster]) -> list[Cluster]:
    """Removes duplicate clusters based on the set of customers, deduplicating within each vehicle type."""
    if not clusters:
        return clusters

    logger.debug(
        f"Starting vehicle-type-aware deduplication with {len(clusters)} clusters."
    )

    # Group clusters by vehicle type
    clusters_by_vehicle_type: Dict[str, List[Cluster]] = {}
    for cluster in clusters:
        vehicle_type = cluster.vehicle_type

        if vehicle_type not in clusters_by_vehicle_type:
            clusters_by_vehicle_type[vehicle_type] = []
        clusters_by_vehicle_type[vehicle_type].append(cluster)

    # Deduplicate within each vehicle type
    unique_clusters = []
    total_duplicates = 0

    for vehicle_type, vehicle_clusters in clusters_by_vehicle_type.items():
        seen_customer_sets = {}
        vehicle_unique = []

        for cluster in vehicle_clusters:
            customer_set = frozenset(cluster.customers)
            if customer_set not in seen_customer_sets:
                seen_customer_sets[customer_set] = cluster
                vehicle_unique.append(cluster)

        duplicates_removed = len(vehicle_clusters) - len(vehicle_unique)
        total_duplicates += duplicates_removed

        if duplicates_removed > 0:
            logger.debug(
                f"Vehicle type {vehicle_type}: Removed {duplicates_removed} duplicate clusters, "
                f"{len(vehicle_unique)} unique clusters remain."
            )

        unique_clusters.extend(vehicle_unique)

    logger.debug(
        f"Finished vehicle-type-aware deduplication: Removed {total_duplicates} duplicate clusters "
        f"across all vehicle types, {len(unique_clusters)} unique clusters remain."
    )

    return unique_clusters


def _get_clustering_context_list(
    params: FleetmixParams,
) -> list[tuple[CapacitatedClusteringContext, str]]:
    """Generates a list of (ClusteringContext, method_name) tuples for all runs."""
    context_list = []

    # Create base context object with common parameters
    base_context = CapacitatedClusteringContext(
        goods=params.problem.goods,
        depot=params.problem.depot,
        max_depth=params.algorithm.clustering_max_depth,
        route_time_estimation=params.algorithm.route_time_estimation,
        geo_weight=params.algorithm.geo_weight,
        demand_weight=params.algorithm.demand_weight,
    )

    method = params.algorithm.clustering_method
    if method == "combine":
        logger.debug("Generating context variations for 'combine' method")

        # Check if sub_methods are specified in the clustering params
        # For now, use default sub_methods since this is not part of the structured params yet
        sub_methods = None
        if sub_methods is None:
            # Use default sub_methods
            sub_methods = ["minibatch_kmeans", "kmedoids", "gaussian_mixture"]

        # 1. Base methods - Use default weights from base_context
        for method_name in sub_methods:
            context_list.append((base_context, method_name))

        # 2. Agglomerative with different explicit weights
        weight_combinations = [
            (1.0, 0.0),
            (0.8, 0.2),
            (0.6, 0.4),
            (0.4, 0.6),
            (0.2, 0.8),
            (0.0, 1.0),
        ]
        for geo_w, demand_w in weight_combinations:
            agglomerative_context = replace(
                base_context, geo_weight=geo_w, demand_weight=demand_w
            )
            context_list.append((agglomerative_context, "agglomerative"))

    else:
        # Single method specified: Use the base_context as configured initially
        logger.debug(f"Using single method configuration: {method}")
        context_list.append((base_context, method))

    logger.debug(
        f"Generated {len(context_list)} distinct clustering context configurations"
    )
    return context_list
