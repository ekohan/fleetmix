from enum import Enum
from pathlib import Path
import time
import pandas as pd

from fleetmix.benchmarking.converters.vrp import convert_vrp_to_fsm
from fleetmix.config.parameters import Parameters
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering import generate_clusters_for_configurations
import fleetmix.optimization as _optimization_module

class VRPType(Enum):
    CVRP = 'cvrp'
    MCVRP = 'mcvrp'


def convert_to_fsm(vrp_type: VRPType, **kwargs) -> tuple[pd.DataFrame, Parameters]:
    """
    Library facade to convert VRP instances to FSM format.
    """
    return convert_vrp_to_fsm(vrp_type, **kwargs)



def run_optimization(
    customers_df: pd.DataFrame,
    params: Parameters,
    verbose: bool = False
) -> tuple[dict, pd.DataFrame]:
    """
    Run the common FSM optimization pipeline.
    Returns the solution dictionary and the configurations DataFrame.
    """
    # Generate vehicle configurations and clusters
    configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
    clusters_df = generate_clusters_for_configurations(
        customers=customers_df,
        configurations_df=configs_df,
        params=params
    )

    solve_fn = _optimization_module.solve_fsm_problem
    solution = solve_fn(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers_df,
        parameters=params,
        verbose=verbose,
    )

    # Console output
    print("\nOptimization Results:")
    print(f"Total Cost: ${solution['total_cost']:,.2f}")
    print(f"Vehicles Used: {sum(solution['vehicles_used'].values())}")
    print(f"Expected Vehicles: {params.expected_vehicles}")

    return solution, configs_df 