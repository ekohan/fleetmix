"""
generator.py

Main module for generating capacity- and time-feasible customer clusters for the fleet design
optimization process. This is the main entry point for the cluster-first phase of the
cluster-first, fleet-design second heuristic.
"""

import itertools
from dataclasses import replace
from multiprocessing import Manager

import pandas as pd
from joblib import Parallel, delayed

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import (
    Cluster,
    ClusteringContext,
    Customer,
    CustomerBase,
    DepotLocation,
    VehicleConfiguration,
)
from fleetmix.merging.core import generate_merge_phase_clusters
from fleetmix.utils.logging import FleetmixLogger
from fleetmix.utils.route_time import estimate_route_time

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


def generate_clusters_for_configurations(
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    params: Parameters,
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

    logger.info("--- Starting Cluster Generation Process ---")
    if not customers or not configurations:
        logger.warning(
            "Input customers or configurations are empty. Returning empty list."
        )
        return []

    with Manager() as manager:
        shared_demand_cache = manager.dict()
        shared_route_time_cache = manager.dict()

        logger.info("Initializing shared caches for demand and route time calculations")

        # 1. Generate feasibility mapping
        logger.info("Generating feasibility mapping...")
        feasible_customers = _generate_feasibility_mapping(
            customers, configurations, params.goods
        )
        if not feasible_customers:
            logger.warning(
                "No customers are feasible for any configuration. Returning empty list."
            )
            return []
        logger.info(
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
            logger.info(
                "TSP route estimation detected. Building distance/duration matrices per vehicle configuration..."
            )
            # Build matrices for each unique avg_speed value across configurations
            from fleetmix.utils.route_time import build_distance_duration_matrices

            unique_speeds = set(config.avg_speed for config in configurations)
            for speed in unique_speeds:
                depot_dict = {
                    "latitude": params.depot.latitude,
                    "longitude": params.depot.longitude,
                }
                # Convert customers to DataFrame for matrix building (temporary)
                customers_df = Customer.to_dataframe(customers)
                build_distance_duration_matrices(customers_df, depot_dict, speed)
                logger.debug(f"Built matrices for avg_speed={speed} km/h")
        else:
            logger.info(
                "TSP route estimation not used. Skipping matrix precomputation."
            )

        cluster_id_generator = itertools.count()

        # 4. Process configurations in parallel for each context configuration
        all_clusters = []
        for clustering_context, method_name in context_and_methods:
            logger.info(
                f"--- Running Configuration: {method_name} (GeoW: {clustering_context.geo_weight:.2f}, DemW: {clustering_context.demand_weight:.2f}) ---"
            )

            # Run clustering for all configurations using these context in parallel, process-based
            clusters_by_config = Parallel(n_jobs=-1, backend="loky")(
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
            logger.info(
                f"--- Configuration {method_name} completed, generated {len([c for config_clusters in clusters_by_config for c in config_clusters])} raw clusters ---"
            )

        logger.info(
            f"Cache statistics: {len(shared_demand_cache)} demand entries, {len(shared_route_time_cache)} route time entries"
        )

    if not all_clusters:
        logger.warning("No clusters were generated by any configuration.")
        return []

    # Remove duplicate clusters based on customer sets
    logger.info(
        f"Combining and deduplicating {len(all_clusters)} raw clusters from all configurations..."
    )
    unique_clusters = _deduplicate_clusters(all_clusters)

    # ------------------------------------------------------------------
    # Merge all pseudo-customers that belong to the same physical customer
    # (identified by Origin_ID) into a single "mega cluster". This gives
    # the optimiser the option of serving the entire site in a single
    # visit (using a multi-compartment vehicle) instead of multiple
    # split-stops.
    # ------------------------------------------------------------------

    if params.allow_split_stops:
        logger.info("🔧 Creating per-origin merged clusters (split-stop helper)…")

        next_id = (
            max(c.cluster_id for c in unique_clusters) + 1 if unique_clusters else 0
        )

        merged = _create_origin_mega_clusters(
            customers,
            configurations,
            params,
            start_cluster_id=next_id,
        )

        if merged:
            logger.info(f"➕ Added {len(merged)} per-origin merged clusters")
            unique_clusters.extend(merged)
            # De-duplicate again in case any customer sets overlap
            unique_clusters = _deduplicate_clusters(unique_clusters)

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
    for good in params.goods:
        base_df[good] = base_df["Config_ID"].map(
            lambda x: config_lookup[str(x)][good] if str(x) in config_lookup else 0
        )
    if "Capacity" not in base_df.columns:
        base_df["Capacity"] = base_df["Config_ID"].map(
            lambda x: config_lookup[str(x)].capacity if str(x) in config_lookup else 0
        )

    # TODO: tidy up parameter hand-off
    pre_params = replace(
        params,
        nearest_merge_candidates=params.pre_nearest_merge_candidates,
        small_cluster_size=params.pre_small_cluster_size,
        post_optimization=False,
    )

    merged_df = generate_merge_phase_clusters(
        selected_clusters=base_df,
        configurations=configurations,
        customers_df=customers_df,
        params=pre_params,
    )

    if not merged_df.empty:
        logger.info(f"➕ Added {len(merged_df)} merged neighbour clusters (pre-MILP)")

        # Ensure unique Cluster_IDs
        next_id = max(c.cluster_id for c in unique_clusters) + 1
        merged_clusters: list[Cluster] = Cluster.from_dataframe(merged_df)
        for cl in merged_clusters:
            cl.cluster_id = next_id
            next_id += 1
        unique_clusters.extend(merged_clusters)

        # Final deduplication
        unique_clusters = _deduplicate_clusters(unique_clusters)

    # Validate cluster coverage
    validate_cluster_coverage(unique_clusters, customers)

    logger.info("--- Cluster Generation Complete ---")
    logger.info(
        f"{Symbols.CHECKMARK} Generated a total of {len(unique_clusters)} unique clusters across all configurations."
    )

    return unique_clusters


def process_configuration(
    config: VehicleConfiguration,
    customers: list[CustomerBase],
    feasible_customers: dict,
    context: ClusteringContext,
    demand_cache: dict | None = None,
    route_time_cache: dict | None = None,
    main_params: Parameters | None = None,
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


def validate_cluster_coverage(clusters: list[Cluster], customers: list[CustomerBase]):
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
        logger.info(
            f"{Symbols.CHECKMARK} All {len(customer_coverage)} customers are covered by at least one cluster."
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
                feasible_configs.append(config.config_id)

        if feasible_configs:
            feasible_customers[customer_id] = feasible_configs

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
    """Removes duplicate clusters based on the set of customers."""
    if not clusters:
        return clusters

    logger.debug(f"Starting deduplication with {len(clusters)} clusters.")

    # Create mapping from customer sets to clusters
    seen_customer_sets = {}
    unique_clusters = []

    for cluster in clusters:
        customer_set = frozenset(cluster.customers)
        if customer_set not in seen_customer_sets:
            seen_customer_sets[customer_set] = cluster
            unique_clusters.append(cluster)

    if len(unique_clusters) < len(clusters):
        logger.debug(
            f"Finished deduplication: Removed {len(clusters) - len(unique_clusters)} duplicate clusters, {len(unique_clusters)} unique clusters remain."
        )
    else:
        logger.debug(
            f"Finished deduplication: No duplicate clusters found ({len(unique_clusters)} clusters)."
        )
    return unique_clusters


def _get_clustering_context_list(
    params: Parameters,
) -> list[tuple[ClusteringContext, str]]:
    """Generates a list of (ClusteringContext, method_name) tuples for all runs."""
    context_list = []

    # Convert depot dict to DepotLocation
    depot_location = DepotLocation(
        latitude=params.depot["latitude"], longitude=params.depot["longitude"]
    )

    # Create base context object with common parameters
    base_context = ClusteringContext(
        goods=params.goods,
        depot=depot_location,
        max_depth=params.clustering["max_depth"],
        route_time_estimation=params.clustering["route_time_estimation"],
        geo_weight=params.clustering["geo_weight"],
        demand_weight=params.clustering["demand_weight"],
    )

    method = params.clustering["method"]
    if method == "combine":
        logger.info("🔄 Generating context variations for 'combine' method")

        # Check if sub_methods are specified in the clustering params
        # TODO: offer this as a parameter
        sub_methods = params.clustering.get("combine_sub_methods", None)
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
        logger.info(f"📍 Using single method configuration: {method}")
        context_list.append((base_context, method))

    logger.info(
        f"Generated {len(context_list)} distinct clustering context configurations."
    )
    return context_list


# ----------------------------------------------------------------------
# Helper: create one cluster per physical customer that merges all its
# pseudo-customers.  This gives the MILP the option to serve all goods in
# one go if a suitable multi-compartment vehicle exists.
# ----------------------------------------------------------------------


# TODO: check this logic
def _create_origin_mega_clusters(
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    params: Parameters,
    start_cluster_id: int = 0,
) -> list[Cluster]:
    """Generate merged clusters per *Origin_ID* (physical customer).

    Only pseudo-customers are considered.  For each physical customer we
    aggregate demand across all its goods subsets.  For every vehicle
    configuration that can carry all those goods and whose capacity and
    route-time constraints are respected we create *one* cluster.
    """

    # Group pseudo-customers by origin_id
    grouped: dict[str, list[CustomerBase]] = {}
    for cust in customers:
        if not cust.is_pseudo_customer():
            continue  # Only relevant for split-stop
        grouped.setdefault(cust.get_origin_id(), []).append(cust)

    if not grouped:
        return []

    clusters: list[Cluster] = []
    cid_counter = start_cluster_id

    for origin_id, pseudo_list in grouped.items():
        # Combine demand vectors and gather coordinates
        total_dem: dict[str, float] = {g: 0.0 for g in params.goods}
        lats, lons = [], []
        for pc in pseudo_list:
            lats.append(pc.location[0])
            lons.append(pc.location[1])
            for g in params.goods:
                total_dem[g] += pc.demands.get(g, 0.0)

        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)

        # Build DataFrame of pseudo customers for route-time estimation
        pseudo_df = Customer.to_dataframe(pseudo_list)

        # Try every configuration; build cluster if feasible
        for cfg in configurations:
            # Check compartments compatibility
            if any(
                total_dem[g] > 0 and not cfg.compartments.get(g, False)
                for g in params.goods
            ):
                continue

            if sum(total_dem.values()) > cfg.capacity:
                continue

            # Route-time estimation
            depot_dict = {
                "latitude": params.depot["latitude"],
                "longitude": params.depot["longitude"],
            }

            # TODO: again, check this logic
            rt_hours, _ = estimate_route_time(
                pseudo_df,
                depot_dict,
                cfg.service_time,
                cfg.avg_speed,
                method=params.clustering["route_time_estimation"],
                max_route_time=cfg.max_route_time,
                prune_tsp=params.prune_tsp,
            )

            if rt_hours > cfg.max_route_time:
                continue

            # Build cluster object
            cluster = Cluster(
                cluster_id=cid_counter,
                config_id=cfg.config_id,
                customers=[pc.customer_id for pc in pseudo_list],
                total_demand=total_dem,
                centroid_latitude=centroid_lat,
                centroid_longitude=centroid_lon,
                goods_in_config=[
                    g for g in params.goods if cfg.compartments.get(g, False)
                ],
                route_time=rt_hours,
                method="origin_merge",
            )

            clusters.append(cluster)
            cid_counter += 1

    return clusters
