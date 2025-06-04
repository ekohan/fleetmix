"""Route time estimation methods for vehicle routing."""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from haversine import haversine

from pyvrp import (
    Model, Client, Depot, VehicleType, ProblemData,
    GeneticAlgorithmParams, PopulationParams, SolveParams
)
from pyvrp.stop import MaxIterations

from fleetmix.utils.logging import FleetmixLogger
from fleetmix.registry import register_route_time_estimator, ROUTE_TIME_ESTIMATOR_REGISTRY
from fleetmix.internal_types import RouteTimeContext, DepotLocation

logger = FleetmixLogger.get_logger(__name__)

def calculate_total_service_time_hours(num_customers: int, service_time_per_customer_minutes: float) -> float:
    """
    Calculates the total service time in hours for a given number of customers
    and a per-customer service time in minutes.

    Args:
        num_customers: The number of customers.
        service_time_per_customer_minutes: Service time for each customer in minutes.

    Returns:
        Total service time in hours.
    """
    if num_customers < 0:
        logger.warning("Number of customers cannot be negative. Returning 0.0 hours service time.")
        return 0.0
    if service_time_per_customer_minutes < 0:
        logger.warning("Service time per customer cannot be negative. Returning 0.0 hours service time.")
        return 0.0
    return (num_customers * service_time_per_customer_minutes) / 60.0

# Global cache for distance and duration matrices (populated if TSP method is used)
_matrix_cache = {
    'distance_matrix': None,
    'duration_matrix': None,
    'customer_id_to_idx': None,
    'depot_idx': 0  # Depot is always at index 0
}

def build_distance_duration_matrices(
    customers_df: pd.DataFrame,
    depot: Dict[str, float],
    avg_speed: float
) -> None:
    """
    Build global distance and duration matrices for all customers plus the depot
    and store them in the module-level cache `_matrix_cache`.

    Args:
        customers_df: DataFrame containing ALL customer data.
        depot: Depot location coordinates {'latitude': float, 'longitude': float}.
        avg_speed: Average vehicle speed (km/h).
    """
    if customers_df.empty:
        logger.warning("Cannot build matrices: Customer DataFrame is empty.")
        return

    logger.info(f"Building global distance/duration matrices for {len(customers_df)} customers...")
    # Create mapping from Customer_ID to matrix index (Depot is 0)
    customer_ids = customers_df['Customer_ID'].tolist()
    # Ensure unique IDs before creating map
    if len(set(customer_ids)) != len(customer_ids):
         logger.warning("Duplicate Customer IDs found. Matrix mapping might be incorrect.")
    customer_id_to_idx = {cid: idx + 1 for idx, cid in enumerate(customer_ids)}

    # Total locations = all customers + depot
    n_locations = len(customers_df) + 1

    # Prepare coordinates list: depot first, then all customers
    depot_coord = (depot['latitude'], depot['longitude'])
    # Ensure Latitude/Longitude columns exist
    if 'Latitude' not in customers_df.columns or 'Longitude' not in customers_df.columns:
        logger.error("Missing 'Latitude' or 'Longitude' columns in customer data.")
        raise ValueError("Missing coordinate columns in customer data.")
        
    customer_coords = list(zip(customers_df['Latitude'], customers_df['Longitude']))
    all_coords = [depot_coord] + customer_coords

    # Initialize matrices
    distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
    duration_matrix = np.zeros((n_locations, n_locations), dtype=int)

    # Speed in km/s for duration calculation
    avg_speed_kps = avg_speed / 3600 if avg_speed > 0 else 0

    # Populate matrices
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            # Distance using Haversine (km)
            dist_km = haversine(all_coords[i], all_coords[j])

            # Store distance in meters (integer)
            distance_matrix[i, j] = distance_matrix[j, i] = int(dist_km * 1000)

            # Duration in seconds
            duration_seconds = (dist_km / avg_speed_kps) if avg_speed_kps > 0 else np.inf # Use infinity if speed is 0
            duration_matrix[i, j] = duration_matrix[j, i] = int(duration_seconds)

    # Update cache
    _matrix_cache['distance_matrix'] = distance_matrix
    _matrix_cache['duration_matrix'] = duration_matrix
    _matrix_cache['customer_id_to_idx'] = customer_id_to_idx
    logger.info(f"Successfully built and cached global matrices ({n_locations}x{n_locations}).")


def estimate_route_time(
    cluster_customers: pd.DataFrame,
    depot: Dict[str, float],
    service_time: float,
    avg_speed: float,
    method: str = 'Legacy',
    max_route_time: float = None,
    prune_tsp: bool = False
) -> Tuple[float, List[str]]:
    """Estimate total route duration for a customer cluster.

    Three alternative heuristics are implemented (select via *method*):

    ``'Legacy'``  – constant 1 h travel + service time component.
    ``'BHH'``     – Beardwood–Halton–Hammersley continuous-space approximation.
    ``'TSP'``     – Solve an exact TSP with *PyVRP* using either cached distance
                    matrices or on-the-fly computation.

    Args:
        cluster_customers: DataFrame with ``Latitude``/``Longitude`` columns and
            a unique ``Customer_ID`` per row.
        depot: Mapping ``{'latitude': float, 'longitude': float}``.
        service_time: Per-customer service time in **minutes**.
        avg_speed: Vehicle speed in **km/h** used to convert distances to time.
        method: One of ``'Legacy'``, ``'BHH'``, ``'TSP'``.
        max_route_time: Optional hard limit (hours) to speed-prune expensive TSP
            evaluations; only relevant when ``method='TSP'``.
        prune_tsp: If *True* and ``method='TSP'`` the BHH estimate is used as a
            quick lower bound to skip TSP calls that are guaranteed infeasible.

    Returns:
        Tuple[float, list[str]]: (estimated route time in **hours**, visit
        sequence).  The sequence is non-empty only for the TSP method; for other
        heuristics an empty list is returned.

    Raises:
        ValueError: If *method* is not recognised.

    Example:
        >>> t, seq = estimate_route_time(cluster, depot, 20, 30, method='BHH')
        >>> t < 8  # hours
        True
    """
    # Look up the estimator from registry
    estimator_class = ROUTE_TIME_ESTIMATOR_REGISTRY.get(method)
    if estimator_class is None:
        raise ValueError(f"Unknown route time estimation method: {method}")
    
    # Create RouteTimeContext object with proper max_route_time handling
    depot_location = DepotLocation(latitude=depot['latitude'], longitude=depot['longitude'])
    # For route time estimation, use a large default if max_route_time is None
    effective_max_route_time = max_route_time if max_route_time is not None else 24 * 7  # 1 week default
    
    context = RouteTimeContext(
        depot=depot_location,
        avg_speed=avg_speed,
        service_time=service_time,
        max_route_time=effective_max_route_time,
        prune_tsp=prune_tsp
    )
    
    # Create instance and estimate
    estimator = estimator_class()
    return estimator.estimate_route_time(cluster_customers, context)


@register_route_time_estimator('Legacy')
class LegacyEstimator:
    """Original simple estimation method."""
    
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> Tuple[float, List[str]]:
        """Legacy estimation: 1 hour travel + service time."""
        num_customers = len(cluster_customers)
        # Convert minutes to hours for service_time component
        time = 1 + calculate_total_service_time_hours(num_customers, context.service_time)
        return time, [] # Return empty sequence


@register_route_time_estimator('BHH')
class BHHEstimator:
    """Beardwood-Halton-Hammersley estimation method."""
    
    # Constants for the BHH formula
    SETUP_TIME = 0.0  # α_vk: Setup time to dispatch vehicle configuration (hours)
    BETA = 0.765      # β: Non-negative constant for BHH approximation
    
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> Tuple[float, List[str]]:
        """BHH estimation: t_vk ≈ α_vk + 2·δ_vk + β·√(n·A) + γ·n
        
        Where:
        - α_vk: Setup time to dispatch vehicle configuration
        - δ_vk: Line-haul travel time between depot and cluster centroid
        - β: Non-negative constant (0.765)
        - γ: Customer service time
        - n: Number of customers
        - A: Service area
        """
        if len(cluster_customers) <= 1:
            # For 0 or 1 customer, service time is the primary component.
            # Assuming service_time is per customer.
            return calculate_total_service_time_hours(len(cluster_customers), context.service_time), []
            
        # Calculate service time component (γ·n)
        service_time_total = calculate_total_service_time_hours(len(cluster_customers), context.service_time)
        
        # Calculate depot travel component (2·δ_vk)
        centroid_lat = cluster_customers['Latitude'].mean()
        centroid_lon = cluster_customers['Longitude'].mean()
        depot_to_centroid = haversine(
            (context.depot.latitude, context.depot.longitude),
            (centroid_lat, centroid_lon)
        )
        depot_travel_time = 2 * depot_to_centroid / context.avg_speed  # Round trip hours
        
        # Calculate intra-cluster travel component using BHH formula (β·√(n·A))
        cluster_radius = max(
            haversine(
                (centroid_lat, centroid_lon),
                (lat, lon)
            )
            for lat, lon in zip(
                cluster_customers['Latitude'],
                cluster_customers['Longitude']
            )
        )
        cluster_area = np.pi * (cluster_radius ** 2)
        intra_cluster_distance = (
            self.BETA *
            np.sqrt(len(cluster_customers)) * 
            np.sqrt(cluster_area)
        )
        intra_cluster_time = intra_cluster_distance / context.avg_speed
        
        # Total time: α_vk + 2·δ_vk + β·√(n·A) + γ·n
        time = self.SETUP_TIME + service_time_total + depot_travel_time + intra_cluster_time
        return time, [] # Return empty sequence


@register_route_time_estimator('TSP')
class TSPEstimator:
    """TSP-based route time estimation using PyVRP."""
    
    def estimate_route_time(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> Tuple[float, List[str]]:
        """Estimate route time by solving a TSP for the cluster using PyVRP."""
        # Prune TSP computation based on BHH estimate if requested
        if context.prune_tsp and context.max_route_time is not None:
            logger.warning(f"Prune TSP: {context.prune_tsp}, Max Route Time: {context.max_route_time}")
            # Use BHH estimator for pruning
            bhh_estimator = BHHEstimator()
            bhh_time, _ = bhh_estimator.estimate_route_time(cluster_customers, context)
            # Add a 20% margin to account for BHH underestimation
            if bhh_time > context.max_route_time * 1.2:
                logger.warning(
                    f"Cluster skipped TSP computation: BHH estimate {bhh_time:.2f}h greatly exceeds max_route_time {context.max_route_time}h"
                )
                return context.max_route_time * 1.01, []  # Slightly over max, empty sequence
        
        # Call the implementation
        return self._pyvrp_tsp_estimation(cluster_customers, context)
    
    def _pyvrp_tsp_estimation(
        self,
        cluster_customers: pd.DataFrame,
        context: RouteTimeContext,
    ) -> Tuple[float, List[str]]:
        """
        Estimate route time by solving a TSP for the cluster using PyVRP.
        Assumes infinite capacity (single vehicle TSP).
        Uses precomputed global matrices from `_matrix_cache` if available.

        Args:
            cluster_customers: DataFrame containing customer data for the cluster.
            context: Route time context including depot, service time, speeds, etc.

        Returns:
            Tuple: (Estimated route time in hours, List of customer IDs in visit sequence or [])
        """
        num_customers = len(cluster_customers)
        
        # Handle edge cases: 0 or 1 customer
        if num_customers == 0:
            return 0.0, []
        if num_customers == 1:
            depot_coord = (context.depot.latitude, context.depot.longitude)
            cust_row = cluster_customers.iloc[0]
            cust_coord = (cust_row['Latitude'], cust_row['Longitude'])
            dist_to = haversine(depot_coord, cust_coord)
            dist_from = haversine(cust_coord, depot_coord)
            travel_time_hours = (dist_to + dist_from) / context.avg_speed
            service_time_hours = calculate_total_service_time_hours(1, context.service_time)
            # Sequence for single customer: Depot -> Customer -> Depot
            sequence = ["Depot", cust_row['Customer_ID'], "Depot"]
            return travel_time_hours + service_time_hours, sequence

        # --- Prepare data for PyVRP TSP ---
        
        # Create mapping from matrix index back to Customer_ID (or "Depot")
        # This needs to be consistent with how matrices are built/sliced
        idx_to_id_map = {}

        # Create PyVRP Depot object (scaling coordinates for precision)
        pyvrp_depot = Depot(
            x=int(context.depot.latitude * 10000), 
            y=int(context.depot.longitude * 10000)
        )

        # Create PyVRP Client objects
        pyvrp_clients = []
        for _, customer in cluster_customers.iterrows():
            pyvrp_clients.append(Client(
                x=int(customer['Latitude'] * 10000),
                y=int(customer['Longitude'] * 10000),
                delivery=[1],  # Dummy demand for TSP
                service_duration=int(context.service_time * 60) # Service time in seconds
            ))

        # Create a single VehicleType with effectively infinite capacity and duration
        # Capacity needs to be at least num_customers for dummy demands
        # Use max_route_time from context
        max_duration_seconds = int(context.max_route_time * 3600)  # Convert hours to seconds
        vehicle_type = VehicleType(
            num_available=1, 
            capacity=[num_customers + 1], # Sufficient capacity for dummy demands
            max_duration=max_duration_seconds # Maximum route time in seconds
        )

        # --- Use sliced matrices from global cache if available, otherwise compute on-the-fly ---
        distance_matrix = None
        duration_matrix = None

        # Check if cache is populated
        cache_ready = (
            _matrix_cache['distance_matrix'] is not None and
            _matrix_cache['duration_matrix'] is not None and
            _matrix_cache['customer_id_to_idx'] is not None
        )

        if cache_ready:
            logger.debug(f"Using cached global matrices for cluster TSP (Size: {num_customers})")
            # Get the global distance and duration matrices and mapping
            global_distance_matrix = _matrix_cache['distance_matrix']
            global_duration_matrix = _matrix_cache['duration_matrix']
            customer_id_to_idx = _matrix_cache['customer_id_to_idx']
            depot_idx = _matrix_cache['depot_idx'] # Should be 0

            # Get indices for this specific cluster (Depot + Cluster Customers)
            cluster_indices = [depot_idx]
            missing_ids = []
            # Map customer IDs to their global indices
            cluster_customer_ids = cluster_customers['Customer_ID'].tolist()
            for customer_id in cluster_customer_ids:
                idx = customer_id_to_idx.get(customer_id)
                if idx is not None:
                    cluster_indices.append(idx)
                else:
                    # This case should ideally not happen if build_distance_duration_matrices
                    # was called with the full customer set. Log a warning if it does.
                    missing_ids.append(customer_id)
                    
            if missing_ids:
                 logger.warning(f"Customer IDs {missing_ids} not found in global matrix cache map. TSP matrix will be incomplete.")
                 # Decide how to handle this - Option 1: Fallback, Option 2: Proceed with warning
                 # Fallback to on-the-fly computation if critical IDs are missing
                 cache_ready = False # Force fallback if any ID is missing

            if cache_ready:
                # Slice the global matrices efficiently using numpy indexing
                n_locations = len(cluster_indices)
                # Use ix_ to select rows and columns based on index list
                distance_matrix = global_distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                duration_matrix = global_duration_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Validate dimensions
                if distance_matrix.shape != (n_locations, n_locations) or \
                   duration_matrix.shape != (n_locations, n_locations):
                     logger.error("Matrix slicing resulted in unexpected dimensions. Fallback needed.")
                     cache_ready = False # Force fallback

                if cache_ready:
                     # Create the index-to-ID map for this specific cluster based on global indices
                     idx_to_id_map[0] = "Depot" # Relative index 0 is always the Depot
                     for i, global_idx in enumerate(cluster_indices[1:], start=1): # Start from relative index 1
                         # Find the customer ID corresponding to this global index
                         # This requires iterating through the global map or having an inverse map
                         for cid, g_idx in customer_id_to_idx.items():
                             if g_idx == global_idx:
                                 idx_to_id_map[i] = cid
                                 break


        # Fallback: Compute matrices on-the-fly if cache wasn't ready or slicing failed
        if not cache_ready:
            logger.debug(f"Cache not ready or slicing failed. Computing matrices on-the-fly for cluster TSP (Size: {num_customers})")
            n_locations = num_customers + 1 # Customers + Depot
            locations_coords = [(context.depot.latitude, context.depot.longitude)] + \
                               list(zip(cluster_customers['Latitude'], cluster_customers['Longitude']))

            distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
            duration_matrix = np.zeros((n_locations, n_locations), dtype=int)

            # Speed in km/s for duration calculation
            avg_speed_kps = context.avg_speed / 3600 if context.avg_speed > 0 else 0

            for i in range(n_locations):
                for j in range(i + 1, n_locations):
                    # Distance using Haversine (km)
                    dist_km = haversine(locations_coords[i], locations_coords[j])

                    # Store distance in meters
                    distance_matrix[i, j] = distance_matrix[j, i] = int(dist_km * 1000)

                    # Duration in seconds
                    duration_seconds = (dist_km / avg_speed_kps) if avg_speed_kps > 0 else np.inf
                    duration_matrix[i, j] = duration_matrix[j, i] = int(duration_seconds)

            # Create the index-to-ID map for this specific cluster
            idx_to_id_map[0] = "Depot" # Index 0 is the Depot
            for i, customer_id in enumerate(cluster_customers['Customer_ID'], start=1): # Start from index 1
                idx_to_id_map[i] = customer_id


        # --- Create Problem Data and Model ---
        # Ensure matrices were actually created (either via cache or on-the-fly)
        if distance_matrix is None or duration_matrix is None:
            logger.error("Distance/Duration matrices could not be obtained for TSP.")
            # Return a large value indicating failure/infeasibility and empty sequence
            return context.max_route_time * 1.1, [] 

        problem_data = ProblemData(
            clients=pyvrp_clients,
            depots=[pyvrp_depot],
            vehicle_types=[vehicle_type],
            distance_matrices=[distance_matrix],
            duration_matrices=[duration_matrix]
        )
        model = Model.from_data(problem_data)

        # --- Solve the TSP using PyVRP's Genetic Algorithm ---
        # Use fewer iterations suitable for smaller TSP instances
        ga_params = GeneticAlgorithmParams(
            repair_probability=0.8, # Standard default
            nb_iter_no_improvement=500 # Reduced iterations
        )
        pop_params = PopulationParams(
            min_pop_size=10, # Smaller population
            generation_size=20,
            nb_elite=2,
            nb_close=3
        )
        # Reduce max iterations for faster solving on small problems
        stop = MaxIterations(max_iterations=1000) 
        
        result = model.solve(
            stop=stop, 
            params=SolveParams(genetic=ga_params, population=pop_params), 
            display=False # No verbose output during estimation
        )
        
        # --- Extract Result ---
        sequence = []
        if result.best.is_feasible():
            # PyVRP duration includes travel and service time in seconds
            total_duration_seconds = result.best.duration() 
            # Extract route sequence - PyVRP returns list of location indices
            # There's only one route in TSP
            if result.best.routes():
                 route_indices = result.best.routes()[0].visits()
                 # Map indices back to Customer IDs using idx_to_id_map
                 # Add Depot at start and end
                 sequence = ["Depot"] + [idx_to_id_map.get(idx, f"UnknownIdx_{idx}") for idx in route_indices] + ["Depot"]
                 logger.debug(f"TSP sequence indices: {route_indices}, mapped: {sequence}")
            else:
                 logger.warning("TSP solution feasible but no route found?")
                 
            # Convert total duration to hours
            return total_duration_seconds / 3600.0, sequence
        else:
            logger.warning(f"TSP solution infeasible for cluster. Returning max time. Num customers: {num_customers}")
            # Return the max route time from context (or slightly higher)
            return context.max_route_time * 1.01, []  # Return slightly over max_route_time and empty sequence