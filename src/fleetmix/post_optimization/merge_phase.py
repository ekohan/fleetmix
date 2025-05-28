"""
merge_phase.py

Implements the **improvement phase** (Section 4.4 of the paper) that iteratively tries to merge
small, neighbouring clusters after the core FSM model has been solved.

Rationale
~~~~~~~~~
The MILP in ``fleetmix.optimization.solve_fsm_problem`` chooses from a *fixed* pool of clusters.  Once an
initial solution is available, additional cost savings can sometimes be obtained by *merging* two
clusters and serving the combined demand with a larger vehicle—provided capacity and route‐time
constraints remain feasible.

Algorithm outline
-----------------
1. Identify "small" clusters (≤ ``params.small_cluster_size`` customers).
2. For each small cluster *s* find the nearest candidate cluster *t* that is:
   • compatible in product mix,  
   • within capacity, and  
   • likely feasible on route‐time (quick lower bound check).
3. Evaluate the merged cluster precisely using :func:`utils.route_time.estimate_route_time` to
   verify that the merged cluster's estimated route time does not exceed the vehicle's maximum route time.
4. Collect all feasible merges, append them to the cluster pool, and re‐optimise the MILP **without
   triggering a recursive improvement**.
5. Repeat until no further cost reduction is achieved or the iteration cap
   ``params.max_improvement_iterations`` is reached.

Caching & performance
---------------------
Route‐time calculations for the same customer sets are memoised in the module‐level dict
``_merged_route_time_cache``.

Outcome
-------
Returns the *best* improved solution dictionary, identical in structure to the one produced by
``fleetmix.optimization.solve_fsm_problem`` but with potentially lower total cost.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from dataclasses import replace

from fleetmix.utils.route_time import estimate_route_time, calculate_total_service_time_hours
from fleetmix.config.parameters import Parameters

from fleetmix.utils.logging import FleetmixLogger, Symbols

logger = FleetmixLogger.get_logger(__name__)

# Cache for merged cluster route times
_merged_route_time_cache: Dict[Tuple[str, ...], Tuple[float, list | None]] = {}

def _get_merged_route_time(
    customers: pd.DataFrame,
    params: Parameters
) -> Tuple[float, list | None]:
    """
    Estimate (and cache) the route time and sequence for a merged cluster of customers.
    Always uses the same method & max_route_time from params.
    """
    key: Tuple[str, ...] = tuple(sorted(customers['Customer_ID']))
    if key in _merged_route_time_cache:
        return _merged_route_time_cache[key]
    
    time, sequence = estimate_route_time(
        cluster_customers=customers,
        depot=params.depot,
        service_time=params.service_time,
        avg_speed=params.avg_speed,
        method=params.clustering['route_time_estimation'],
        max_route_time=params.max_route_time,
        prune_tsp=params.prune_tsp
    )
    _merged_route_time_cache[key] = (time, sequence)
    return time, sequence

def improve_solution(
    initial_solution: Dict,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Dict:
    """Iteratively merge small clusters to lower total cost.

    Implements the *improvement phase* described in Section 4.4.  Starting from
    an existing solution dictionary, the algorithm searches for pairs of
    "small" clusters that can be feasibly served together by the same vehicle
    configuration and whose merge reduces the overall objective value.

    Args:
        initial_solution: Solution dictionary returned by
            :func:`fleetmix.optimization.solve_fsm_problem`.
        configurations_df: Vehicle configuration catalogue (same as used in the
            optimisation step).
        customers_df: Original customer dataframe; required for route-time
            recalculation and centroid updates when evaluating merges.
        params: Parameter object controlling thresholds such as
            ``small_cluster_size``, ``max_improvement_iterations``, etc.

    Returns:
        Dict: Improved solution dictionary (same schema as *initial_solution*).
        If no improving merge is found the original dictionary is returned.

    Example:
        >>> improved = improve_solution(sol, configs, customers, params)
        >>> improved['total_cost'] <= sol['total_cost']
        True
    """
    from fleetmix.optimization import solve_fsm_problem

    best = initial_solution
    best_cost = best.get('total_cost', float('inf'))
    reason = ''
    # Iterate with explicit counter to correctly log attempts
    for iters in range(1, params.max_improvement_iterations + 1):
        logger.debug(f"\n{Symbols.CHECK} Merge phase iteration {iters}/{params.max_improvement_iterations}")
        selected_clusters = best.get('selected_clusters', best.get('clusters'))
        if selected_clusters is None:
            logger.error("Cannot find clusters in solution.")
            reason = "no clusters"
            break

        # Ensure goods columns exist
        for good in params.goods:
            if good not in selected_clusters.columns:
                selected_clusters[good] = selected_clusters['Config_ID'].map(
                    lambda x: configurations_df[configurations_df['Config_ID'] == x].iloc[0][good]
                )
        merged_clusters = generate_merge_phase_clusters(
            selected_clusters,
            configurations_df,
            customers_df,
            params
        )
        if merged_clusters.empty:
            logger.debug("→ No valid merged clusters generated")
            reason = "no candidate merges"
            break

        logger.debug(f"→ Generated {len(merged_clusters)} merged cluster options")
        combined_clusters = pd.concat([selected_clusters, merged_clusters], ignore_index=True)
        # Call solver without triggering another merge phase
        internal_params = replace(params, post_optimization=False)
        trial = solve_fsm_problem(
            combined_clusters,
            configurations_df,
            customers_df,
            internal_params
        )
        trial_cost = trial.get('total_cost', float('inf'))
        cost_better = trial_cost < best_cost - 1e-6

        same_choice = False
        if 'selected_clusters' in trial and 'selected_clusters' in best:
            trial_ids = set(trial['selected_clusters']['Cluster_ID'])
            best_ids  = set(best['selected_clusters']['Cluster_ID'])
            same_choice = (trial_ids == best_ids)

        logger.debug(f"→ Trial cost={trial_cost:.2f}, best cost={best_cost:.2f}, Δ={(trial_cost-best_cost):.2f}")
        if not cost_better:
            reason = "no cost improvement"
            break
        if same_choice:
            reason = "same chosen clusters"
            break

        # Accept improvement and continue
        best = trial
        best_cost = trial_cost
    else:
        # Loop completed without breaks
        reason = "iteration cap reached"
        iters = params.max_improvement_iterations

    logger.debug(f"Merge phase finished after {iters} iteration(s): {reason}")
    return best

def generate_merge_phase_clusters(
    selected_clusters: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    params: Parameters
) -> pd.DataFrame:
    """Generate merged clusters from selected small clusters."""
    new_clusters = []
    stats = {
        'attempted': 0, 
        'valid': 0, 
        'invalid_time': 0, 
        'invalid_capacity': 0,
        'invalid_compatibility': 0
    }
    
    # Create an indexed DataFrame for efficient configuration lookups
    configs_indexed = configurations_df.set_index('Config_ID')
    # Index customers for fast lookup
    customers_indexed = customers_df.set_index('Customer_ID')
    
    # Start from selected_clusters (which already has goods columns) and add capacity
    cluster_meta = selected_clusters.copy()
    cluster_meta['Capacity'] = cluster_meta['Config_ID'].map(configs_indexed['Capacity'])
    small_meta = cluster_meta[
        cluster_meta['Customers'].apply(len) <= params.small_cluster_size
    ]
    if small_meta.empty:
        return pd.DataFrame()
    target_meta = cluster_meta

    logger.debug(f"→ Found {len(small_meta)} small clusters")

    # Precompute numpy arrays for vectorized capacity & goods checks
    goods_arr = target_meta[params.goods].to_numpy()
    cap_arr   = target_meta['Capacity'].to_numpy()
    ids       = target_meta['Cluster_ID'].to_numpy()
    lat_arr   = target_meta['Centroid_Latitude'].to_numpy()
    lon_arr   = target_meta['Centroid_Longitude'].to_numpy()

    # Vectorized filtering loop
    for _, small in small_meta.iterrows():
        sd = np.array([small['Total_Demand'][g] for g in params.goods])
        peak = sd.max()
        needs = sd > 0
        if needs.any():
            goods_ok = (goods_arr[:, needs] == 1).all(axis=1)
        else:
            goods_ok = np.ones_like(cap_arr, dtype=bool)
        cap_ok   = cap_arr >= peak
        not_self = ids != small['Cluster_ID']

        # Proximity-based filtering: compute distances and pick nearest candidates
        small_point = (small['Centroid_Latitude'], small['Centroid_Longitude'])
        target_points = np.column_stack((lat_arr, lon_arr))
        
        distances = haversine_vector(small_point, target_points, unit=Unit.KILOMETERS, comb=True)
        distances = distances.flatten() # Ensure distances is a 1D array
        
        valid_mask = cap_ok & goods_ok & not_self & ~np.isnan(distances)
        valid_idxs = np.where(valid_mask)[0]
        if valid_idxs.size == 0:
            continue
        nearest_idxs = valid_idxs[np.argsort(distances[valid_idxs])[:params.nearest_merge_candidates]]
        for idx in nearest_idxs:
            # Quick lower-bound time prune before costly route-time estimation
            target = target_meta.iloc[idx]
            rt_target = target['Route_Time']
            rt_small  = small['Route_Time']

            # Compute service time for the cluster not contributing the max route_time (avoid double count)
            if rt_small > rt_target:
                svc_time_other = calculate_total_service_time_hours(len(target['Customers']), params.service_time)
            else:
                svc_time_other = calculate_total_service_time_hours(len(small['Customers']), params.service_time)

            # Quick lower-bound time prune before costly route-time estimation (no proximity term)
            lb = max(rt_target, rt_small) + svc_time_other
            if lb > params.max_route_time:
                stats['invalid_time'] = stats.get('invalid_time', 0) + 1
                logger.debug(f"Lower-bound prune: merge {small['Cluster_ID']} + {target['Cluster_ID']} lb={lb:.2f} > max={params.max_route_time:.2f}")
                continue
            stats['attempted'] += 1
            target_config = configs_indexed.loc[target['Config_ID']]
            is_valid, route_time, demands, tsp_sequence = validate_merged_cluster(
                small, target, target_config, customers_indexed, params
            )
            if not is_valid:
                # Assuming validate_merged_cluster now logs reasons for invalidity if needed
                # or updates stats for invalid_capacity, invalid_compatibility
                continue
            stats['valid'] += 1

            # Build merged cluster
            merged_customer_ids = target['Customers'] + small['Customers']
            # Ensure merged_customers are fetched correctly for centroid calculation
            # It's crucial that customers_indexed contains all relevant customers.
            # If validate_merged_cluster already did this, we might optimize, but safety first.
            current_merged_customers_df = customers_indexed.loc[merged_customer_ids].reset_index()

            centroid_lat = current_merged_customers_df['Latitude'].mean()
            centroid_lon = current_merged_customers_df['Longitude'].mean()
            new_cluster = {
                'Cluster_ID': f"{target['Cluster_ID']}_{small['Cluster_ID']}",
                'Config_ID': target['Config_ID'],
                'Customers': merged_customer_ids,
                'Route_Time': route_time,
                'Total_Demand': demands,
                'Method': f"merged_{target['Method']}"
            ,
                'Centroid_Latitude': centroid_lat,
                'Centroid_Longitude': centroid_lon
            }
            if tsp_sequence is not None:
                new_cluster['TSP_Sequence'] = tsp_sequence
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
        'Cluster_ID', 'Config_ID', 'Customers', 'Route_Time', 
        'Total_Demand', 'Method', 'Centroid_Latitude', 'Centroid_Longitude', 'TSP_Sequence'
    ] + list(params.goods)
    
    # Build and dedupe merged clusters
    df = pd.DataFrame(new_clusters, columns=minimal_columns)
    return df.drop_duplicates('Cluster_ID')

def validate_merged_cluster(
    cluster1: pd.Series,
    cluster2: pd.Series,
    config: pd.Series,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Tuple[bool, float, Dict, list | None]:
    """Validate if two clusters can be merged."""
    # Index customers for fast lookup
    if customers_df.index.name != 'Customer_ID':
        customers_indexed = customers_df.set_index('Customer_ID', drop=False)
    else:
        customers_indexed = customers_df
    # Check compartment compatibility
    merged_goods = {}
    for g in params.goods:
        # Handle case where Total_Demand might be a dict or series
        demand1 = cluster1['Total_Demand'][g] if isinstance(cluster1['Total_Demand'], (dict, pd.Series)) else cluster1[g]
        demand2 = cluster2['Total_Demand'][g] if isinstance(cluster2['Total_Demand'], (dict, pd.Series)) else cluster2[g]
        merged_goods[g] = demand1 + demand2
    
    # Validate capacity
    if any(demand > config['Capacity'] for demand in merged_goods.values()):
        return False, 0, {}, None

    # Get all customers from both clusters
    cluster1_customers = cluster1['Customers'] if isinstance(cluster1['Customers'], list) else [cluster1['Customers']]
    cluster2_customers = cluster2['Customers'] if isinstance(cluster2['Customers'], list) else [cluster2['Customers']]
    
    merged_customers_ids = cluster1_customers + cluster2_customers
    # Validate that all customer IDs are present in customers_indexed
    # Check if customers_indexed has 'Customer_ID' as its index
    if customers_indexed.index.name != 'Customer_ID':
        # This case should ideally not happen if indexing is consistent
        logger.error("customers_indexed is not indexed by 'Customer_ID' in validate_merged_cluster.")
        # Fallback or raise error, for now, assume it's an issue and return invalid
        return False, 0, {}, None

    missing_ids = [cid for cid in merged_customers_ids if cid not in customers_indexed.index]
    if missing_ids:
        logger.warning(f"Missing customer IDs {missing_ids} during merge validation for potential merge of clusters involving {cluster1.get('Cluster_ID', 'Unknown')} and {cluster2.get('Cluster_ID', 'Unknown')}.")
        return False, 0, {}, None

    merged_customers = customers_indexed.loc[
        merged_customers_ids
    ].reset_index()
    
    # Validate customer locations
    if (merged_customers['Latitude'].isna().any() or 
        merged_customers['Longitude'].isna().any()):
        return False, 0, {}, None

    # Estimate (and cache) new route time using the general estimator
    new_route_time, new_sequence = _get_merged_route_time(merged_customers, params)

    if new_route_time > params.max_route_time:
        return False, 0, {}, None

    return True, new_route_time, merged_goods, new_sequence 