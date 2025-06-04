from __future__ import annotations

from enum import Enum
from pathlib import Path
import pandas as pd

from fleetmix.benchmarking.converters.vrp import convert_vrp_to_fsm
from fleetmix.config.parameters import Parameters
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.optimization import solve_fsm_problem
from fleetmix.utils.logging import log_progress, log_success, log_detail
from fleetmix.utils.time_measurement import TimeRecorder
from fleetmix.core_types import FleetmixSolution, VehicleConfiguration

class VRPType(Enum):
    CVRP = 'cvrp'
    MCVRP = 'mcvrp'


def vehicle_configurations_to_dataframe(configs: list[VehicleConfiguration]) -> pd.DataFrame:
    """Convert list of VehicleConfiguration to DataFrame for compatibility."""
    return pd.DataFrame([config.to_dict() for config in configs])


def convert_to_fsm(vrp_type: VRPType, **kwargs) -> tuple[pd.DataFrame, Parameters]:
    """
    Library facade to convert VRP instances to FSM format.
    """
    return convert_vrp_to_fsm(vrp_type, **kwargs)


def run_optimization(
    customers_df: pd.DataFrame,
    params: Parameters,
    verbose: bool = False
) -> tuple[FleetmixSolution, pd.DataFrame]:
    """
    Run the common FSM optimization pipeline.
    Returns the solution dictionary and the configurations DataFrame.
    """
    # Initialize TimeRecorder
    time_recorder = TimeRecorder()
    
    with time_recorder.measure("global"):
        # Generate vehicle configurations and clusters
        with time_recorder.measure("vehicle_configuration"):
            configs = generate_vehicle_configurations(params.vehicles, params.goods)
        
        with time_recorder.measure("clustering"):
            clusters_df = generate_clusters_for_configurations(
                customers=customers_df,
                configurations=configs,
                params=params
            )

        with time_recorder.measure("fsm_initial"):
            solution = solve_fsm_problem(
                clusters_df=clusters_df,
                configurations=configs,
                customers_df=customers_df,
                parameters=params,
                verbose=verbose,
                time_recorder=time_recorder
            )

    # Add time measurements to solution
    solution.time_measurements = time_recorder.measurements

    # Console output
    log_progress("Optimization Results:")
    log_detail(f"Total Cost: ${solution.total_cost:,.2f}")
    log_detail(f"Vehicles Used: {sum(solution.vehicles_used.values())}")
    log_detail(f"Expected Vehicles: {params.expected_vehicles}")

    # Convert configs to DataFrame for return (for save_optimization_results compatibility)
    configs_df = vehicle_configurations_to_dataframe(configs)
    return solution, configs_df 