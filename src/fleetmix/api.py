"""
API facade for Fleetmix - provides a single entry point for programmatic usage.
"""
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import pandas as pd

from fleetmix.config.parameters import Parameters
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.optimization import solve_fsm_problem

logger = logging.getLogger(__name__)


def optimize(
    demand: Union[str, Path, pd.DataFrame],
    config: Optional[Union[str, Path, Parameters]] = None,
    output_dir: str = "results",
    format: str = "excel",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the Fleetmix optimization pipeline.
    
    Args:
        demand: Path to CSV file, Path object, or pandas DataFrame with customer demand data
        config: Path to YAML config file, Path object, or Parameters object (optional)
        output_dir: Directory where results will be saved (default: "results")
        format: Output format - "excel" or "json" (default: "excel")
        verbose: Enable verbose output (default: False)
        
    Returns:
        Dictionary containing the optimization solution with keys:
        - total_fixed_cost: Total fixed cost of selected vehicles
        - total_variable_cost: Total variable/routing cost
        - total_penalties: Total penalties (light load + compartment)
        - vehicles_used: List of vehicles used in the solution
        - selected_clusters: DataFrame of selected clusters
        - missing_customers: List of customers not served
        - solver_status: Status of the optimization solver
        - solver_runtime_sec: Time taken by solver
        - post_optimization_runtime_sec: Time for post-optimization
        
    Raises:
        FileNotFoundError: If demand or config file not found
        ValueError: If optimization is infeasible or configuration is invalid
        Exception: For unexpected errors with original details
    """
    try:
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
            configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
        except Exception as e:
            raise ValueError(
                f"Error generating vehicle configurations:\n{str(e)}\n"
                f"Please check your vehicle and goods definitions in the config."
            )
        
        # Step 4: Generate clusters
        try:
            clusters_df = generate_clusters_for_configurations(
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
            solution = solve_fsm_problem(
                clusters_df=clusters_df,
                configurations_df=configs_df,
                customers_df=customers,
                parameters=params,
                verbose=verbose
            )
            
            # Check if optimization was successful
            if solution['solver_status'] != 'Optimal':
                if 'infeasible' in solution['solver_status'].lower():
                    # Analyze why it's infeasible
                    missing_count = len(solution.get('missing_customers', []))
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
                    logger.warning(f"Solver returned non-optimal status: {solution['solver_status']}")
                    
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
                    execution_time=solution.get('total_runtime_sec', 0),
                    solver_name=solution['solver_name'],
                    solver_status=solution['solver_status'],
                    solver_runtime_sec=solution['solver_runtime_sec'],
                    post_optimization_runtime_sec=solution['post_optimization_runtime_sec'],
                    configurations_df=configs_df,
                    selected_clusters=solution['selected_clusters'],
                    total_fixed_cost=solution['total_fixed_cost'],
                    total_variable_cost=solution['total_variable_cost'],
                    total_light_load_penalties=solution['total_light_load_penalties'],
                    total_compartment_penalties=solution['total_compartment_penalties'],
                    total_penalties=solution['total_penalties'],
                    vehicles_used=solution['vehicles_used'],
                    missing_customers=solution['missing_customers'],
                    parameters=params,
                    format=format
                )
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
                # Don't fail the entire operation if saving fails
                
        return solution
        
    except (FileNotFoundError, ValueError):
        # Re-raise our custom errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        raise Exception(
            f"Unexpected error during optimization:\n{str(e)}\n"
            f"Error type: {type(e).__name__}\n"
            f"Please check the logs for more details."
        ) from e 