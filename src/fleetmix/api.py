"""
API facade for Fleetmix - provides a single entry point for programmatic usage.
"""
from pathlib import Path
from typing import Union, Optional, Dict, Any
import pandas as pd
import time

from fleetmix.config.parameters import Parameters
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.clustering import generate_feasible_clusters
from fleetmix.optimization import optimize_fleet_selection
from fleetmix.utils.logging import FleetmixLogger, log_warning
from fleetmix.utils.time_measurement import TimeRecorder
from fleetmix.internal_types import FleetmixSolution as _InternalSolution
from fleetmix.types import (
    FleetmixSolution,
    ClusterAssignment,
    VehicleConfiguration,
)

logger = FleetmixLogger.get_logger('fleetmix.api')


def _to_public_solution(sol: _InternalSolution, params: Parameters) -> FleetmixSolution:
    """Convert internal solution to public API solution."""
    # Extract unique configurations used
    config_ids = sol.selected_clusters['Config_ID'].unique()
    configs_df = sol.selected_clusters[['Config_ID', 'Vehicle_Type', 'Capacity', 'Fixed_Cost']].drop_duplicates()
    
    # Build configuration objects
    configurations = []
    for _, row in configs_df.iterrows():
        # Get compartments from the selected clusters for this config
        config_clusters = sol.selected_clusters[sol.selected_clusters['Config_ID'] == row['Config_ID']]
        first_cluster = config_clusters.iloc[0]
        
        compartments = {g: bool(first_cluster.get(g, 0)) for g in params.goods}
        
        # Get the vehicle spec from params to get operational parameters
        vehicle_type = row['Vehicle_Type']
        vehicle_spec = None
        for vname, vspec in params.vehicles.items():
            if vname == vehicle_type:
                vehicle_spec = vspec
                break
        
        if vehicle_spec is None:
            # Fallback to defaults if vehicle not found
            avg_speed = 30.0
            service_time = 25.0
            max_route_time = 10.0
        else:
            avg_speed = vehicle_spec.avg_speed
            service_time = vehicle_spec.service_time
            max_route_time = vehicle_spec.max_route_time
        
        configurations.append(
            VehicleConfiguration(
                config_id=int(row['Config_ID']),
                vehicle_type=row['Vehicle_Type'],
                compartments=compartments,
                capacity=int(row['Capacity']),
                fixed_cost=float(row['Fixed_Cost']),
                avg_speed=avg_speed,
                service_time=service_time,
                max_route_time=max_route_time,
            )
        )
    
    # Build cluster assignments
    clusters = []
    for _, row in sol.selected_clusters.iterrows():
        clusters.append(
            ClusterAssignment(
                cluster_id=int(row['Cluster_ID']),
                config_id=int(row['Config_ID']),
                customer_ids=list(row['Customers']),
                route_time=float(row['Route_Time']),
                total_demand=dict(row['Total_Demand']),
                centroid=(float(row['Centroid_Latitude']), float(row['Centroid_Longitude'])),
            )
        )
    
    return FleetmixSolution(
        selected_clusters=clusters,
        configurations_used=configurations,
        total_cost=float(sol.total_cost),
        total_vehicles=int(sol.total_vehicles),
        missing_customers=set(sol.missing_customers),
        solver_status=str(sol.solver_status),
        solver_runtime_sec=float(sol.solver_runtime_sec),
        # Additional fields
        total_fixed_cost=float(sol.total_fixed_cost),
        total_variable_cost=float(sol.total_variable_cost),
        total_penalties=float(sol.total_penalties),
        vehicles_used=dict(sol.vehicles_used) if sol.vehicles_used else None,
    )


def optimize(
    demand: Union[str, Path, pd.DataFrame],
    config: Optional[Union[str, Path, Parameters]] = None,
    output_dir: str = "results",
    format: str = "excel",
    verbose: bool = False
) -> FleetmixSolution:
    """
    Run the Fleetmix optimization pipeline.
    
    Args:
        demand: Path to CSV file, Path object, or pandas DataFrame with customer demand data
        config: Path to YAML config file, Path object, or Parameters object (optional)
        output_dir: Directory where results will be saved (default: "results")
        format: Output format - "excel" or "json" (default: "excel")
        verbose: Enable verbose output (default: False)
        
    Returns:
        FleetmixSolution: The optimization solution containing:
        - selected_clusters: List of ClusterAssignment objects
        - configurations_used: List of VehicleConfiguration objects used
        - total_cost: Total cost of the solution
        - total_vehicles: Number of vehicles used
        - missing_customers: Set of customer IDs not served
        - solver_status: Status of the optimization solver
        - solver_runtime_sec: Time taken by solver
        
    Raises:
        FileNotFoundError: If demand or config file not found
        ValueError: If optimization is infeasible or configuration is invalid
        Exception: For unexpected errors with original details
    """
    try:

        time_recorder = TimeRecorder()
        with time_recorder.measure("global"):
            
            # Step 1: Load customer demand
            if isinstance(demand, pd.DataFrame):
                # Convert DataFrame to expected format
                # The DataFrame should have columns: Customer_ID, Latitude, Longitude, and demand columns
                customers = demand.copy()
                
                # Ensure required columns exist
                required_cols = ['Customer_ID', 'Latitude', 'Longitude']
                missing_cols = [col for col in required_cols if col not in customers.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Also add Customer_Name column if it doesn't exist
                if 'Customer_Name' not in customers.columns:
                    customers['Customer_Name'] = customers['Customer_ID'].astype(str)
                
                # Add any missing demand columns with 0
                for good in ['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']:
                    if good not in customers.columns:
                        customers[good] = 0
                        
            else:
                demand_path = Path(demand)
                if not demand_path.exists():
                    raise FileNotFoundError(
                        f"Demand file not found: {demand_path}\n"
                        f"Please check the file path and ensure it exists."
                    )
                
                # Try to load the CSV directly first to check its format
                try:
                    df = pd.read_csv(demand_path)
                    
                    # Check if it's already in wide format (has demand columns)
                    demand_cols = ['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']
                    if all(col in df.columns for col in demand_cols):
                        # Already in wide format, use it directly
                        customers = df
                        # Ensure Customer_Name column exists
                        if 'Customer_Name' not in customers.columns:
                            customers['Customer_Name'] = customers['Customer_ID'].astype(str)
                    else:
                        # It's in the long format expected by load_customer_demand
                        customers = load_customer_demand(str(demand_path))
                        
                except Exception as e:
                    # Fall back to load_customer_demand 
                    customers = load_customer_demand(str(demand_path))
            
            # Step 2: Load parameters
            if config is None:
                # Use default parameters by loading from default config file
                params = Parameters.from_yaml()
            elif isinstance(config, Parameters):
                params = config
            else:
                config_path = Path(config)
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_path}\n"
                        f"Please check the file path and ensure it exists."
                    )
                try:
                    params = Parameters.from_yaml(str(config_path))
                except Exception as e:
                    raise ValueError(
                        f"Error loading configuration from {config_path}:\n{str(e)}\n"
                        f"Please check the YAML syntax and required fields."
                    )
            
            # Step 3: Generate vehicle configurations
            try:
                with time_recorder.measure("vehicle_configuration"):
                    from fleetmix.utils.vehicle_configurations import _generate_vehicle_configurations_df
                    configs_df = _generate_vehicle_configurations_df(params.vehicles, params.goods)
            except Exception as e:
                raise ValueError(
                    f"Error generating vehicle configurations:\n{str(e)}\n"
                    f"Please check your vehicle and goods definitions in the config."
                )
            
            # Step 4: Generate clusters
            try:
                with time_recorder.measure("clustering"):
                    from fleetmix.clustering.generator import _generate_feasible_clusters_df
                    clusters_df = _generate_feasible_clusters_df(
                        customers=customers,
                        configurations_df=configs_df,
                        params=params
                    )
                
                if len(clusters_df) == 0:
                    raise ValueError(
                        "No feasible clusters could be generated!\n"
                        "Possible causes:\n"
                        "- Vehicle capacities are too small for customer demands\n"
                        "- Time windows are too restrictive\n" 
                        "- Service times + travel times exceed time limits\n"
                        "Please review your configuration and customer data."
                    )
                    
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Error generating clusters:\n{str(e)}\n"
                    f"Please check your clustering parameters."
                )
            
            # Step 5: Solve optimization problem
            try:
                with time_recorder.measure("fsm_initial"):
                    solution = optimize_fleet_selection(
                        clusters_df=clusters_df,
                        configurations_df=configs_df,
                        customers_df=customers,
                        parameters=params,
                        verbose=verbose,
                        time_recorder=time_recorder
                    )
                
                solution.time_measurements = time_recorder.measurements

                # Check if optimization was successful
                if solution.solver_status != 'Optimal':
                    if 'infeasible' in solution.solver_status.lower():
                        # Analyze why it's infeasible
                        missing_count = len(solution.missing_customers)
                        total_customers = len(customers)
                        
                        error_msg = (
                            f"Optimization problem is infeasible!\n"
                            f"The solver could not find a valid solution.\n"
                        )
                        
                        if missing_count > 0:
                            error_msg += (
                                f"\nMissing customers: {missing_count}/{total_customers}\n"
                                f"Possible solutions:\n"
                                f"- Increase max_vehicles in config\n"
                                f"- Add more vehicle types\n"
                                f"- Increase vehicle capacities\n"
                                f"- Relax time window constraints\n"
                                f"- Reduce service times"
                            )
                        else:
                            error_msg += (
                                f"\nAll customers can be served, but other constraints are violated.\n"
                                f"Check your cost parameters and penalties."
                            )
                            
                        raise ValueError(error_msg)
                    else:
                        log_warning(f"Solver returned non-optimal status: {solution.solver_status}")
                    
            except ValueError:
                raise
            except Exception as e:
                raise Exception(
                    f"Error during optimization:\n{str(e)}\n"
                    f"This may be due to solver issues or invalid problem formulation."
                )
            
            # Step 6: Save results if requested
            if output_dir:
                try:
                    save_optimization_results(
                        solution=solution, # Pass the whole solution object
                        configurations_df=configs_df,
                        parameters=params,
                        format=format
                    )
                except Exception as e:
                    log_warning(f"Failed to save results: {str(e)}")
                    # Don't fail the entire operation if saving fails
                    
            # Convert to public API solution before returning
            return _to_public_solution(solution, params)
        
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        raise Exception(
            f"Unexpected error during optimization:\n{str(e)}\n"
            f"Error type: {type(e).__name__}\n"
            f"Please check the logs for more details."
        ) from e 