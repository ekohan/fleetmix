"""
Single-compartment VRP solver module using PyVRP.
Provides baseline comparison for multi-compartment vehicle solutions.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pyvrp import (
    Model, GeneticAlgorithmParams,
    Depot, Client, VehicleType, ProblemData,
    PopulationParams, SolveParams
)
from pyvrp.stop import MaxIterations
from haversine import haversine
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fleetmix.config.parameters import Parameters
from fleetmix.utils.logging import (
    FleetmixLogger,
    log_progress, log_detail, log_debug,
    log_error, Colors, Symbols
)
from fleetmix.utils.route_time import estimate_route_time
from fleetmix.core_types import BenchmarkType, VRPSolution
logger = FleetmixLogger.get_logger(__name__)

class VRPSolver:
    """Single-compartment VRP solver implementation."""
    
    def __init__(
        self,
        customers: pd.DataFrame,
        params: Parameters,
        time_limit: int = 300,
        benchmark_type: BenchmarkType = BenchmarkType.SINGLE_COMPARTMENT
    ):
        self.customers = customers
        self.params = params
        self.time_limit = time_limit
        self.benchmark_type = benchmark_type
        self.route_time_estimation = params.clustering['route_time_estimation']
        self.model = self._prepare_model()
    
    def _prepare_model(self) -> Model:
        """Prepare PyVRP model from customer data."""
        expanded_clients = []
        
        if self.benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            # For single product solving, create a client for each product demand
            for _, row in self.customers.iterrows():
                for good in self.params.goods:
                    demand = row[f'{good}_Demand']
                    if demand > 0:
                        expanded_clients.append(Client(
                            x=int(row['Latitude'] * 10000),
                            y=int(row['Longitude'] * 10000),
                            delivery=[int(demand)],
                            service_duration=self.params.service_time * 60
                        ))
        else:
            # For multi-compartment, use total demand
            for _, row in self.customers.iterrows():
                total_demand = sum(row[f'{good}_Demand'] for good in self.params.goods)
                if total_demand > 0:
                    expanded_clients.append(Client(
                        x=int(row['Latitude'] * 10000),
                        y=int(row['Longitude'] * 10000),
                        delivery=[int(total_demand)],
                        service_duration=self.params.service_time * 60
                    ))

        # Create vehicle types with proper capacities
        vehicle_types = [
            VehicleType(
                num_available=len(expanded_clients),
                capacity=[vt_info['capacity']],
                fixed_cost=vt_info['fixed_cost'],
                max_duration=self.params.max_route_time * 3600
            )
            for vt_name, vt_info in self.params.vehicles.items()
        ]

        # Calculate base distance matrix
        base_distance_matrix = self._calculate_distance_matrix(len(expanded_clients))
        duration_matrix = (base_distance_matrix / self.params.avg_speed) * 3600

        # Create problem data
        self.data = ProblemData(
            clients=expanded_clients,
            depots=[Depot(
                x=int(self.params.depot['latitude'] * 10000),
                y=int(self.params.depot['longitude'] * 10000)
            )],
            vehicle_types=vehicle_types,
            distance_matrices=[base_distance_matrix],
            duration_matrices=[duration_matrix]
        )

        return Model.from_data(self.data)
    
    def _calculate_distance_matrix(self, n_clients: int) -> np.ndarray:
        """Calculate distance matrix for expanded client list."""
        distance_matrix = np.zeros((n_clients + 1, n_clients + 1))  # +1 for depot
        client_coords_list = [(0, 0)] * (n_clients + 1) # Store coords (lat, lon) for depot + clients
        
        # Add depot coordinates at index 0
        depot_coords = (self.params.depot['latitude'], self.params.depot['longitude'])
        client_coords_list[0] = depot_coords
        
        # Calculate distances and store client coordinates
        client_idx = 0
        for _, row in self.customers.iterrows():
            coords = (row['Latitude'], row['Longitude'])
            
            # Check if this customer should be included based on benchmark type
            if self.benchmark_type == BenchmarkType.MULTI_COMPARTMENT:
                total_demand = sum(row[f'{good}_Demand'] for good in self.params.goods)
                if total_demand > 0:
                    client_idx += 1
                    dist = haversine(depot_coords, coords)
                    distance_matrix[0, client_idx] = dist
                    distance_matrix[client_idx, 0] = dist
                    client_coords_list[client_idx] = coords # Store coords
            else:  # SINGLE_COMPARTMENT
                for good in self.params.goods:
                    if row[f'{good}_Demand'] > 0:
                        client_idx += 1
                        dist = haversine(depot_coords, coords)
                        distance_matrix[0, client_idx] = dist
                        distance_matrix[client_idx, 0] = dist
                        client_coords_list[client_idx] = coords # Store coords
        
        # Calculate client-to-client distances using stored coordinates
        for i in range(1, n_clients + 1):
            for j in range(i + 1, n_clients + 1):
                coords_i = client_coords_list[i]
                coords_j = client_coords_list[j]
                dist = haversine(coords_i, coords_j)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def solve_scv_parallel(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """Solve VRP instances for each product type in parallel."""
        # Split customers by product type
        product_instances = []
        for good in self.params.goods:
            mask = self.customers[f'{good}_Demand'] > 0
            if mask.any():
                # Create a copy with only this product's demand
                product_customers = self.customers.copy()
                # Zero out other products' demands
                for other_good in self.params.goods:
                    if other_good != good:
                        product_customers[f'{other_good}_Demand'] = 0
                product_instances.append((good, product_customers))
        
        # Solve in parallel using existing solve method
        solutions = Parallel(n_jobs=-1)(
            delayed(self._solve_single_product)(customers, verbose)
            for _, customers in product_instances
        )
        
        return {
            product: solution 
            for (product, _), solution in zip(product_instances, solutions)
        }
    
    def _solve_single_product(self, customers: pd.DataFrame, verbose: bool) -> VRPSolution:
        """Helper method to solve a single product instance."""
        solver = VRPSolver(
            customers=customers,
            params=self.params,
            time_limit=self.time_limit,
            benchmark_type=BenchmarkType.SINGLE_COMPARTMENT
        )
        return solver.solve_scv(verbose=verbose)
    
    def solve_scv(self, verbose: bool = False) -> VRPSolution:
        """Solve the VRP instance."""
        # Create genetic algorithm parameters with balanced values
        ga_params = GeneticAlgorithmParams(
            repair_probability=0.9,
            nb_iter_no_improvement=10000  # Increased from 2000
        )
        
        # Create population parameters with larger population for better exploration
        pop_params = PopulationParams(
            min_pop_size=40,  # Increased from 25
            generation_size=60,  # Increased from 40
            nb_elite=6,  # Increased from 4
            nb_close=8,  # Increased from 5
            lb_diversity=0.1,
            ub_diversity=0.6  # Increased from 0.5
        )
        
        # Create stopping criterion with more iterations
        stop = MaxIterations(max_iterations=5000)  # Increased from 2000
        
        # Solve and return best solution
        result = self.model.solve(
            stop=stop,
            params=SolveParams(genetic=ga_params, population=pop_params),
            display=verbose
        )
        
        # Extract solution details
        solution = result.best
        
        # Get routes (keep as PyVRP Route objects)
        routes = solution.routes()
        
        # Calculate route times and check feasibility
        feasible_routes = []
        route_times = []
        route_feasibility = []  # Track feasibility of each route
        
        for route_idx, route in enumerate(routes):
            if len(route) <= 1:
                if verbose:
                    print(f"Route {route_idx} skipped: Empty route")
                continue

            # Get total demand for this route
            total_demand = sum(
                self.data.clients()[client - 1].delivery[0] 
                for i in range(1, len(route)-1)
                if (client := route[i]) > 0
            )
            
            # Get vehicle type and capacity
            vehicle_type_idx = route.vehicle_type()
            vehicle_type = self.data.vehicle_types()[vehicle_type_idx]
            vehicle_capacity = vehicle_type.capacity[0]
            
            # Calculate utilization percentage
            utilization = (total_demand / vehicle_capacity) * 100
            
            # Create DataFrame of route customers for time estimation
            route_customers = pd.DataFrame([
                {
                    'Latitude': self.data.clients()[client - 1].x / 10000,
                    'Longitude': self.data.clients()[client - 1].y / 10000
                }
                for i in range(1, len(route)-1)
                if (client := route[i]) > 0
            ])
            
            # Calculate route time
            route_time, _ = estimate_route_time(
                cluster_customers=route_customers,
                depot=self.params.depot,
                service_time=self.params.service_time,
                avg_speed=self.params.avg_speed,
                method='BHH',
                max_route_time=self.params.max_route_time,
                prune_tsp=self.params.prune_tsp
            )
            
            # Check if route is feasible (but include it anyway)
            is_feasible = (utilization <= 100 and route_time <= self.params.max_route_time)
            
            # Log route status
            if not is_feasible:
                if utilization > 100:
                    logger.warning(f"{Colors.RED}Route {route_idx} exceeds capacity (Utilization: {utilization:.1f}%){Colors.RESET}")
                if route_time > self.params.max_route_time:
                    logger.warning(f"{Colors.RED}Route {route_idx} exceeds max time ({route_time:.2f} > {self.params.max_route_time}){Colors.RESET}")
            elif verbose:
                logger.info(f"{Colors.GREEN}Route {route_idx} feasible: Utilization={utilization:.1f}%, Time={route_time:.2f}h{Colors.RESET}")
            
            feasible_routes.append([route[i] for i in range(len(route))])
            route_times.append(route_time)
            route_feasibility.append(is_feasible)
        
        if not feasible_routes:
            return VRPSolution(
                total_cost=float('inf'),
                fixed_cost=0.0,
                variable_cost=0.0,
                total_distance=float('inf'),
                num_vehicles=0,
                routes=[],
                vehicle_loads=[],
                execution_time=result.runtime,
                solver_status="Infeasible",
                route_sequences=[],
                vehicle_utilization=[],
                vehicle_types=[],
                route_times=[],
                route_distances=[],
                route_feasibility=[]
            )
            
        # Use feasible_routes for solution
        routes = feasible_routes
        
        # Calculate costs
        fixed_cost = solution.fixed_vehicle_cost()
        variable_cost = solution.duration_cost() + solution.distance_cost()
        
        # Calculate vehicle utilization correctly for each route
        vehicle_utilizations = []
        vehicle_types = []
        vehicle_loads = []
        route_distances = []
        
        # Process routes before converting to lists
        for route_idx, pyvrp_route in enumerate(solution.routes()):
            if pyvrp_route.visits():  # Skip empty routes
                vehicle_type_idx = pyvrp_route.vehicle_type()
                vehicle_type = self.data.vehicle_types()[vehicle_type_idx]
                vehicle_capacity = vehicle_type.capacity[0]

                # Calculate total load for this route
                load = sum(self.data.clients()[i-1].delivery[0] for i in pyvrp_route.visits())
                
                # Ensure load does not exceed capacity
                if load > vehicle_capacity:
                    logger.warning(f"Route {route_idx} exceeds vehicle capacity: {load}/{vehicle_capacity}")
                    load = vehicle_capacity

                vehicle_loads.append(load)
                vehicle_utilizations.append((load / vehicle_capacity) * 100)  # Store as percentage
                vehicle_types.append(vehicle_type_idx)  # Store the vehicle type index
                route_distances.append(solution.distance())
                
        # Now convert routes to lists for storage
        routes = [[i for i in route] for route in solution.routes()]

        return VRPSolution(
            total_cost=fixed_cost + variable_cost,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            total_distance=solution.distance(),
            num_vehicles=len(feasible_routes),
            routes=feasible_routes,
            vehicle_loads=vehicle_loads,
            execution_time=result.runtime,
            solver_status="Optimal" if solution.is_feasible() else "Infeasible",
            route_sequences=[[str(i) for i in route] for route in feasible_routes],
            vehicle_utilization=vehicle_utilizations,
            vehicle_types=vehicle_types,
            route_times=route_times,
            route_distances=route_distances,
            route_feasibility=route_feasibility  # Add the feasibility information
        )
    
    def _print_solution(
        self,
        total_cost: float,
        total_distance: float,
        num_vehicles: int,
        routes: List[List[int]],
        execution_time: float,
        utilization: List[float],
        benchmark_type: BenchmarkType,
        compartment_configs: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """Print solution details."""
        log_progress("🚚 VRP Solution Summary:")
        
        # Handle infeasible solutions
        if total_cost == float('inf') or not routes:
            log_error("Status: INFEASIBLE")
            log_error("No feasible solution found")
            log_detail(f"Execution Time: {execution_time:.1f}s")
            return

        # Print solution details for feasible solutions
        log_detail(f"Total Cost: ${total_cost:,.2f}")
        log_detail(f"Total Distance: {total_distance:.1f} km")
        log_detail(f"Total Vehicles Used: {num_vehicles}")
        
        # Only calculate utilization if we have valid routes
        if utilization:
            log_detail(f"Avg Vehicle Utilization: {np.mean(utilization)*100:.1f}%")
        log_detail(f"Execution Time: {execution_time:.1f}s")

        if benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            # Count vehicles by product type
            vehicles_by_product = {good: 0 for good in self.params.goods}
            for route in routes:
                if route:  # If route is not empty
                    # Assuming client_products is a list mapping client index to product type
                    product = self.client_products[route[1]-1]  # Get product type of first client in route (skip depot)
                    vehicles_by_product[product] += 1
            
            # Print vehicle breakdown
            log_detail("Vehicles by Product Type:")
            for product, count in vehicles_by_product.items():
                log_detail(f"→ {product}: {count}")

        # Print compartment configurations if available
        if compartment_configs:
            log_detail("Compartment Configurations:")
            for i, config in enumerate(compartment_configs, 1):
                # Calculate total used capacity
                total_used_capacity = sum(config.values())
                # Calculate empty capacity
                empty_capacity = 1.0 - total_used_capacity
                
                log_detail(f"Route {i}:")
                for product, percentage in config.items():
                    if percentage >= 0.01:  # Only show if >= 1%
                        log_detail(f"  {product}: {percentage*100:.1f}%")
                
                # Always print empty capacity
                log_detail(f"  Empty: {empty_capacity*100:.1f}%")
    
    def _prepare_multi_compartment_data(self) -> pd.DataFrame:
        """
        Prepare customer data for multi-compartment solving by aggregating demands.
        Also stores original demand breakdown for post-processing.
        """
        # Create a copy of customer data
        mc_customers = self.customers.copy()
        
        # Store original demands per product type for later use
        self.original_demands = {}
        
        # Use index as customer ID if CustomerID column doesn't exist
        id_column = 'CustomerID' if 'CustomerID' in self.customers.columns else self.customers.index
        
        for idx, row in self.customers.iterrows():
            customer_id = str(row[id_column]) if 'CustomerID' in self.customers.columns else str(idx)
            self.original_demands[customer_id] = {
                good: row.get(f'{good}_Demand', 0)
                for good in self.params.goods
            }
        
        # Calculate total demand for each customer
        mc_customers['Total_Demand'] = mc_customers.apply(
            lambda row: sum(row.get(f'{good}_Demand', 0) for good in self.params.goods),
            axis=1
        )
        
        return mc_customers

    def _determine_compartment_configuration(
        self,
        route_customers: List[str],
        vehicle_type_idx: int
    ) -> Dict[str, float]:
        """
        Determine optimal compartment configuration for a route.
        Returns dict mapping product types to their required capacity percentage.
        """
        # Calculate total demand per product type for this route
        route_demands = {good: 0.0 for good in self.params.goods}
        
        # Get the vehicle type from the solution
        vehicle_type = list(self.params.vehicles.values())[vehicle_type_idx]
        total_vehicle_capacity = vehicle_type['capacity']
        
        for customer_id in route_customers:
            for good in self.params.goods:
                route_demands[good] += self.original_demands[customer_id][good]
        
        # Calculate percentage for each product type relative to vehicle capacity
        compartments = {
            good: demand / total_vehicle_capacity
            for good, demand in route_demands.items()
            if demand > 0
        }
        
        return compartments

    def solve_mcv(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """Solve multi-compartment VRP instance."""
        # Prepare aggregated data
        mc_customers = self._prepare_multi_compartment_data()
        
        # Create and solve VRP with aggregated demands
        solver = VRPSolver(
            customers=mc_customers,
            params=self.params,
            time_limit=self.time_limit,
            benchmark_type=BenchmarkType.MULTI_COMPARTMENT
        )
        
        # Get base solution
        base_solution = solver.solve_scv(verbose=verbose)
        
        # Filter valid routes and their indices
        valid_indices = [i for i, r in enumerate(base_solution.routes) if r and len(r) > 2]
        
        # Calculate compartment configurations for valid routes
        route_configurations = []
        for idx in valid_indices:
            route = base_solution.routes[idx]
            route_customers = [str(mc_customers.index[client - 1]) for client in route[1:]]
            compartments = self._determine_compartment_configuration(
                route_customers,
                base_solution.vehicle_types[idx]
            )
            route_configurations.append(compartments)
        
        # Create filtered solution
        mc_solution = VRPSolution(
            total_cost=base_solution.total_cost,
            fixed_cost=base_solution.fixed_cost,
            variable_cost=base_solution.variable_cost,
            total_distance=base_solution.total_distance,
            num_vehicles=len(valid_indices),
            routes=[base_solution.routes[i] for i in valid_indices],
            vehicle_loads=[base_solution.vehicle_loads[i] for i in valid_indices],
            execution_time=base_solution.execution_time,
            solver_status=base_solution.solver_status,
            route_sequences=[base_solution.route_sequences[i] for i in valid_indices],
            vehicle_utilization=[base_solution.vehicle_utilization[i] for i in valid_indices],
            vehicle_types=[base_solution.vehicle_types[i] for i in valid_indices],
            route_times=[base_solution.route_times[i] for i in valid_indices],
            route_distances=[base_solution.route_distances[i] for i in valid_indices],
            route_feasibility=[base_solution.route_feasibility[i] for i in valid_indices]
        )
        
        if verbose:
            self._print_solution(
                total_cost=mc_solution.total_cost,
                total_distance=mc_solution.total_distance,
                num_vehicles=mc_solution.num_vehicles,
                routes=mc_solution.routes,
                execution_time=mc_solution.execution_time,
                utilization=mc_solution.vehicle_utilization,
                benchmark_type=BenchmarkType.MULTI_COMPARTMENT,
                compartment_configs=route_configurations
            )
        
        return {"multi": mc_solution, "compartment_configs": route_configurations}
    
    def _print_diagnostic_information(self, customers: pd.DataFrame) -> None:
        """Print diagnostic information about the input data."""
        log_debug(f"{Symbols.INFO} VRP Solver Diagnostic Information:")
        
        # Count customers by product type
        customers_by_product = {}
        for good in self.params.goods:
            demand_col = f'{good}_Demand'
            if demand_col in customers.columns:
                count = customers[customers[demand_col] > 0].shape[0]
                customers_by_product[good] = count
        
        # Print customer counts
        log_debug("Customers by Product Type:")
        for product, count in customers_by_product.items():
            log_debug(f"→ {product}: {count}")
        
        # Calculate total demand by product type
        demand_by_product = {}
        for good in self.params.goods:
            demand_col = f'{good}_Demand'
            if demand_col in customers.columns:
                total_demand = customers[demand_col].sum()
                demand_by_product[good] = total_demand
        
        # Print demand information
        log_debug("Total Demand by Product Type:")
        for product, demand in demand_by_product.items():
            log_debug(f"→ {product}: {demand:.1f}")
        
        # Calculate minimum vehicles needed (assuming single compartment)
        min_vehicles_by_product = {}
        for good, demand in demand_by_product.items():
            # Get vehicle with smallest capacity that can handle this product
            compatible_vehicles = []
            for vehicle_name, vehicle_info in self.params.vehicles.items():
                compartments = vehicle_info.get('compartments', {})
                if compartments.get(good, False):
                    compatible_vehicles.append((vehicle_name, vehicle_info))
            
            if compatible_vehicles:
                # Sort by capacity
                compatible_vehicles.sort(key=lambda x: x[1]['capacity'])
                smallest_vehicle = compatible_vehicles[0]
                capacity = smallest_vehicle[1]['capacity']
                min_vehicles = np.ceil(demand / capacity)
                min_vehicles_by_product[good] = min_vehicles
        
        # Print minimum vehicles needed
        log_debug("Minimum Vehicles Needed (Single Compartment):")
        for product, count in min_vehicles_by_product.items():
            log_debug(f"→ {product}: {count:.0f}")
        
        # Print total for all products
        total_vehicles = sum(min_vehicles_by_product.values())
        log_debug(f"→ Total: {total_vehicles:.0f}")
        
        # Print expected vehicle count
        if hasattr(self.params, 'expected_vehicles'):
            log_debug(f"Expected Vehicles: {self.params.expected_vehicles}")
        
    def solve(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """Solve VRP using appropriate strategy based on benchmark type."""
        if verbose:
            self._print_diagnostic_information(self.customers)
            
        if self.benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            # Solve single-compartment VRP with separate vehicles for each product
            result = self.solve_scv_parallel(verbose=verbose)
            
            # Calculate combined metrics
            total_cost = sum(s.total_cost for s in result.values())
            total_distance = sum(s.total_distance for s in result.values())
            total_vehicles = sum(s.num_vehicles for s in result.values())
            
            if verbose:
                log_detail(f"{Symbols.SUCCESS} Combined Results:")
                log_detail(f"→ Total Cost: ${total_cost:,.2f}")
                log_detail(f"→ Total Distance: {total_distance:.1f} km")
                log_detail(f"→ Total Vehicles: {total_vehicles}")
            
            return result
        else:
            # Solve multi-compartment VRP
            return self.solve_mcv(verbose=verbose) 