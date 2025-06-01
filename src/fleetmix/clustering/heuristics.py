"""
heuristics.py

This module implements the clustering heuristics algorithms used in the cluster generation process.
It contains the lower-level implementation details of the clustering algorithms, constraint checking,
and recursive splitting logic.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from kmedoids import KMedoids
from sklearn.mixture import GaussianMixture
from fleetmix.config.parameters import Parameters
from fleetmix.utils.route_time import estimate_route_time
from fleetmix.registry import register_clusterer, CLUSTERER_REGISTRY

from .common import Cluster, ClusteringSettings
from fleetmix.utils.logging import FleetmixLogger
logger = FleetmixLogger.get_logger(__name__)

# Product weights for demand profile calculations - equal weighting for all product types
PRODUCT_WEIGHTS = {
    'Frozen': 1.0 / 3.0,    # Equal priority (1/3)
    'Chilled': 1.0 / 3.0,   # Equal priority (1/3)
    'Dry': 1.0 / 3.0        # Equal priority (1/3)
}


@register_clusterer('minibatch_kmeans')
class MiniBatchKMeansClusterer:
    """MiniBatch KMeans clustering algorithm."""
    
    def fit(self, customers: pd.DataFrame, *, settings: ClusteringSettings, n_clusters: int) -> List[int]:
        """Cluster customers using MiniBatch KMeans."""
        data = compute_cluster_metric_input(customers, settings)
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        return model.fit_predict(data).tolist()


@register_clusterer('kmedoids')
class KMedoidsClusterer:
    """K-Medoids clustering algorithm."""
    
    def fit(self, customers: pd.DataFrame, *, settings: ClusteringSettings, n_clusters: int) -> List[int]:
        """Cluster customers using K-Medoids."""
        data = compute_cluster_metric_input(customers, settings)
        model = KMedoids(
            n_clusters=n_clusters,
            metric='euclidean',
            method='fasterpam',
            init='build',
            max_iter=300,
            random_state=42
        )
        return model.fit_predict(data).tolist()


@register_clusterer('agglomerative')
class AgglomerativeClusterer:
    """Agglomerative clustering algorithm."""
    
    def fit(self, customers: pd.DataFrame, *, settings: ClusteringSettings, n_clusters: int) -> List[int]:
        """Cluster customers using Agglomerative clustering."""
        # Agglomerative clustering needs precomputed distance matrix
        data = compute_cluster_metric_input(customers, settings)
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        return model.fit_predict(data).tolist()


@register_clusterer('gaussian_mixture')
class GaussianMixtureClusterer:
    """Gaussian Mixture Model clustering algorithm."""
    
    def fit(self, customers: pd.DataFrame, *, settings: ClusteringSettings, n_clusters: int) -> List[int]:
        """Cluster customers using Gaussian Mixture Model."""
        data = compute_cluster_metric_input(customers, settings)
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            covariance_type='full'
        )
        return model.fit_predict(data).tolist()


def compute_cluster_metric_input(
    customers: pd.DataFrame,
    settings: ClusteringSettings
) -> np.ndarray:
    """Get appropriate input for clustering algorithm."""
    # Methods that need precomputed distance matrix
    needs_precomputed = (
        settings.method.startswith('agglomerative') 
    )
    
    if needs_precomputed:
        logger.debug(f"Using precomputed distance matrix for method: {settings.method} with geo_weight={settings.geo_weight}, demand_weight={settings.demand_weight}")
        return compute_composite_distance(customers, settings.goods, settings.geo_weight, settings.demand_weight)
    else:
        logger.debug(f"Using feature-based input for method: {settings.method}")
        # Ensure data is in the right format - contiguous float64 array
        data = customers[['Latitude', 'Longitude']].values
        return np.ascontiguousarray(data, dtype=np.float64)

def compute_composite_distance(
    customers: pd.DataFrame,
    goods: List[str],
    geo_weight: float,
    demand_weight: float
) -> np.ndarray:
    """Compute composite distance matrix combining geographical and demand distances."""
    # Compute geographical distance
    coords = customers[['Latitude', 'Longitude']].values
    geo_dist = pairwise_distances(coords, metric='euclidean')
    
    # Compute demand profiles
    demands = customers[[f'{g}_Demand' for g in goods]].fillna(0).values
    demand_profiles = np.zeros_like(demands, dtype=float)
    
    # Convert to proportions
    total_demands = demands.sum(axis=1)
    nonzero_mask = total_demands > 0
    for i in range(len(goods)):
        demand_profiles[nonzero_mask, i] = demands[nonzero_mask, i] / total_demands[nonzero_mask]
    
    # Apply temperature sensitivity weights
    for i, good in enumerate(goods):
        demand_profiles[:, i] *= PRODUCT_WEIGHTS.get(good, 1.0)
    
    # Compute demand similarity using cosine distance
    demand_dist = pairwise_distances(demand_profiles, metric='cosine')
    demand_dist = np.nan_to_num(demand_dist, nan=1.0)
    
    # Normalize distances
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()
    
    # Combine distances with weights
    composite_distance = (geo_weight * geo_dist) + (demand_weight * demand_dist)
    
    return composite_distance

def get_cached_demand(
    customers: pd.DataFrame,
    goods: List[str],
    demand_cache: Dict
) -> Dict[str, float]:
    """Get demand from cache or compute and cache it."""
    # Use sorted tuple of customer IDs as key (immutable and hashable)
    key = tuple(sorted(customers['Customer_ID']))
    
    # Check if in cache
    result = demand_cache.get(key, None)
    if result is not None:
        return result
    
    # Not in cache, compute it
    demand_dict = {g: float(customers[f'{g}_Demand'].sum()) for g in goods}
    
    # Store in cache
    demand_cache[key] = demand_dict
    return demand_dict

def get_cached_route_time(
    customers: pd.DataFrame,
    settings: ClusteringSettings,
    route_time_cache: Dict,
    params: Parameters
) -> Tuple[float, List[str]]:
    """Get route time and sequence (if TSP) from cache or compute and cache it."""
    key = tuple(sorted(customers['Customer_ID']))
    result = route_time_cache.get(key, None)
    if result is not None:
        return result
    
    route_time, route_sequence = estimate_route_time(
        cluster_customers=customers,
        depot=settings.depot,
        service_time=settings.service_time,
        avg_speed=settings.avg_speed,
        method=settings.route_time_estimation,
        max_route_time=settings.max_route_time,
        prune_tsp=params.prune_tsp
    )
    
    route_time_cache[key] = (route_time, route_sequence)
    return route_time, route_sequence

def get_feasible_customers_subset(
    customers: pd.DataFrame, 
    feasible_customers: Dict, 
    config_id: int
) -> pd.DataFrame:
    """Extract feasible customers for a given configuration."""
    return customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

def create_initial_clusters(
    customers_subset: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings,
    params: Parameters
) -> pd.DataFrame:
    """Create initial clusters for the given customer subset."""
    # Create a working copy
    customers_copy = customers_subset.copy()
    
    # Add total demand directly if needed for the algorithm
    # This is equivalent to what add_demand_information was doing
    customers_copy['Total_Demand'] = customers_copy[[f'{g}_Demand' for g in settings.goods]].sum(axis=1)
    
    if len(customers_copy) <= 2:
        return create_small_dataset_clusters(customers_copy)
    else:
        return create_normal_dataset_clusters(customers_copy, config, settings, params)

def create_small_dataset_clusters(customers_subset: pd.DataFrame) -> pd.DataFrame:
    """Create clusters for small datasets (≤2 customers)."""
    customers_copy = customers_subset.copy()
    data = customers_copy[['Latitude', 'Longitude']].values
    data = np.ascontiguousarray(data, dtype=np.float64)
    model = MiniBatchKMeans(n_clusters=1, random_state=42)
    customers_copy['Cluster'] = model.fit_predict(data)
    return customers_copy

def create_normal_dataset_clusters(
    customers_subset: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings,
    params: Parameters
) -> pd.DataFrame:
    """Create clusters for normal-sized datasets."""
    customers_copy = customers_subset.copy()
    
    # Determine number of clusters
    num_clusters = estimate_num_initial_clusters(
        customers_copy,
        config,
        settings,
        params
    )
    
    # Ensure the number of clusters doesn't exceed the number of customers
    num_clusters = min(num_clusters, len(customers_copy))
    
    # Get the clusterer from registry
    clusterer_class = CLUSTERER_REGISTRY.get(settings.method)
    if clusterer_class is None:
        logger.error(f"❌ Unknown clustering method: {settings.method}")
        raise ValueError(f"Unknown clustering method: {settings.method}")
    
    # Create instance and cluster
    clusterer = clusterer_class()
    labels = clusterer.fit(customers_copy, settings=settings, n_clusters=num_clusters)
    customers_copy['Cluster'] = labels
    
    return customers_copy

def generate_cluster_id_base(config_id: int) -> int:
    """Generate a base cluster ID from the configuration ID."""
    return int(str(config_id) + "000")

def check_constraints(
    cluster_customers: pd.DataFrame,
    config: pd.Series,
    settings: ClusteringSettings,
    demand_cache: Dict,
    route_time_cache: Dict,
    params: Parameters
) -> tuple[bool, bool]:
    """
    Check if cluster violates capacity or time constraints.
    
    Returns:
        tuple: (capacity_violated, time_violated)
    """
    # Get demand from cache
    demand_dict = get_cached_demand(
        cluster_customers, 
        settings.goods, 
        demand_cache
    )
    cluster_demand = sum(demand_dict.values())
    
    # Get route time from cache (ignore sequence for constraint check)
    route_time, _ = get_cached_route_time(
        cluster_customers,
        settings,
        route_time_cache,
        params
    )
    
    capacity_violated = cluster_demand > config['Capacity']
    time_violated = route_time > settings.max_route_time
    
    return capacity_violated, time_violated

def should_split_cluster(
    cluster_customers: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings, 
    depth: int,
    demand_cache: Dict,
    route_time_cache: Dict,
    params: Parameters
) -> bool:
    """Determine if a cluster should be split based on constraints."""
    capacity_violated, time_violated = check_constraints(
        cluster_customers, 
        config, 
        settings,
        demand_cache,
        route_time_cache,
        params
    )
    is_singleton_cluster = len(cluster_customers) <= 1
    
    # Log warning for single-customer constraints
    if (capacity_violated or time_violated) and is_singleton_cluster:
        logger.debug(f"⚠️ Can't split further (singleton cluster) but constraints still violated: "
                     f"capacity={capacity_violated}, time={time_violated}")
    
    # Return True if we need to split (constraints violated and we can split)
    return (capacity_violated or time_violated) and not is_singleton_cluster

def split_cluster(
    cluster_customers: pd.DataFrame, 
    settings: ClusteringSettings
) -> List[pd.DataFrame]:
    """Split an oversized cluster into smaller ones."""
    # Get the clusterer from registry
    clusterer_class = CLUSTERER_REGISTRY.get(settings.method)
    if clusterer_class is None:
        logger.error(f"❌ Unknown clustering method: {settings.method}")
        raise ValueError(f"Unknown clustering method: {settings.method}")
    
    # Create instance and split into 2 clusters
    clusterer = clusterer_class()
    sub_labels = clusterer.fit(cluster_customers, settings=settings, n_clusters=2)
    
    # Convert list to numpy array for indexing
    sub_labels = np.array(sub_labels)
    
    # Create sub-clusters
    sub_clusters = []
    sub_cluster_sizes = []
    for label in [0, 1]:
        sub_cluster = cluster_customers[sub_labels == label]
        if not sub_cluster.empty:
            sub_clusters.append(sub_cluster)
            sub_cluster_sizes.append(len(sub_cluster))
    
    logger.debug(f"Split cluster of size {len(cluster_customers)} into {len(sub_clusters)} "
                f"sub-clusters of sizes {sub_cluster_sizes}")
    
    return sub_clusters

def create_cluster(
    cluster_customers: pd.DataFrame, 
    config: pd.Series, 
    cluster_id: int, 
    settings: ClusteringSettings,
    demand_cache: Dict,
    route_time_cache: Dict,
    params: Parameters
) -> Cluster:
    """Create a Cluster object from customer data."""
    # Get demand from cache
    total_demand = get_cached_demand(
        cluster_customers, 
        settings.goods, 
        demand_cache
    )
    
    # Get route time and sequence from cache
    route_time, tsp_sequence = get_cached_route_time(
        cluster_customers,
        settings,
        route_time_cache,
        params
    )
    
    cluster = Cluster(
        cluster_id=cluster_id,
        config_id=config['Config_ID'],
        customers=cluster_customers['Customer_ID'].tolist(),
        total_demand=total_demand,
        centroid_latitude=float(cluster_customers['Latitude'].mean()),
        centroid_longitude=float(cluster_customers['Longitude'].mean()),
        goods_in_config=[g for g in settings.goods if config[g] == 1],
        route_time=route_time,
        method=settings.method,
        tsp_sequence=tsp_sequence
    )
    return cluster

def process_clusters_recursively(
    initial_clusters_df: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings,
    demand_cache: Dict,
    route_time_cache: Dict,
    params: Parameters = None
) -> List[Cluster]:
    """Process clusters recursively to ensure constraints are satisfied."""
    config_id = config['Config_ID']
    cluster_id_base = generate_cluster_id_base(config_id)
    current_cluster_id = 0
    clusters = [] 
    
    # Process clusters until all constraints are satisfied
    clusters_to_check = [
        (initial_clusters_df[initial_clusters_df['Cluster'] == c], 0)
        for c in initial_clusters_df['Cluster'].unique()
    ]
    
    logger.info(f"Starting recursive processing for config {config_id} with {len(clusters_to_check)} initial clusters")
    
    split_count = 0
    skipped_count = 0
    
    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        max_depth_reached = depth >= settings.max_depth
        
        # Check if max depth reached
        if max_depth_reached:
            # Check if constraints violated
            capacity_violated, time_violated = check_constraints(
                cluster_customers, 
                config, 
                settings,
                demand_cache,
                route_time_cache,
                params
            )
            
            if capacity_violated or time_violated:
                logger.debug(f"⚠️ Max depth {settings.max_depth} reached but constraints still violated: "
                              f"capacity={capacity_violated}, time={time_violated}, "
                              f"method={settings.method}, config_id={config['Config_ID']}")
                skipped_count += 1
                continue  # Skip this cluster
        
        # Not at max depth, check if we should split
        if not max_depth_reached and should_split_cluster(cluster_customers, config, settings, depth, demand_cache, route_time_cache, params):
            split_count += 1
            logger.debug(f"Splitting cluster for config {config_id} (size {len(cluster_customers)}) at depth {depth}/{settings.max_depth}")
            # Split oversized clusters
            for sub_cluster in split_cluster(cluster_customers, settings):
                clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster (either constraints satisfied or could not be split further)
            current_cluster_id += 1
            cluster = create_cluster(
                cluster_customers, 
                config,
                cluster_id_base + current_cluster_id, 
                settings,
                demand_cache,
                route_time_cache,
                params
            )
            clusters.append(cluster)
    
    if skipped_count > 0:
        logger.debug(f"⚠️ Skipped {skipped_count} clusters that exceeded capacity at max depth for config {config_id}.")
    
    logger.info(f"Completed recursive processing for config {config_id}: {len(clusters)} final clusters, "
               f"{split_count} splits performed")
    
    return clusters

def estimate_num_initial_clusters(
    customers: pd.DataFrame,
    config: pd.Series,
    settings: ClusteringSettings,
    params: Parameters = None
) -> int:
    """
    Estimate the number of initial clusters needed based on capacity and time constraints.
    """
    if customers.empty:
        return 0

    # Default prune_tsp flag if params not provided
    prune_tsp_val = params.prune_tsp if params is not None else False

    # Calculate total demand for relevant goods
    total_demand = 0
    for good in settings.goods:
        if config[good]:  # Only consider goods this vehicle can carry
            total_demand += customers[f'{good}_Demand'].sum()

    # Estimate clusters needed based on capacity
    clusters_by_capacity = np.ceil(total_demand / config['Capacity'])

    # Estimate time for an average route
    avg_customers_per_cluster = len(customers) / clusters_by_capacity
    # Ensure sample size doesn't exceed population size and is at least 1 if possible
    sample_size = max(1, min(int(avg_customers_per_cluster), len(customers)))
    avg_cluster = customers.sample(n=sample_size)
    # Unpack the tuple returned by estimate_route_time
    avg_route_time, _ = estimate_route_time(
        cluster_customers=avg_cluster,
        depot=settings.depot,
        service_time=settings.service_time, # in minutes
        avg_speed=settings.avg_speed,
        method=settings.route_time_estimation, # Use the specific method for estimation
        max_route_time=settings.max_route_time, # Pass max_route_time
        prune_tsp=prune_tsp_val # Use params.prune_tsp or default False
    )

    # Estimate clusters needed based on time
    clusters_by_time = np.ceil(
        avg_route_time * len(customers) / 
        (settings.max_route_time * avg_customers_per_cluster)
    )

    # Take the maximum of the two estimates
    num_clusters = int(max(clusters_by_capacity, clusters_by_time, 1))
    
    logger.debug(
        f"Estimated clusters: {num_clusters} "
        f"(capacity: {clusters_by_capacity}, time: {clusters_by_time})"
    )
    
    return num_clusters 