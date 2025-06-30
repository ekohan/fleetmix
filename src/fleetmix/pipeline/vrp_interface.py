from __future__ import annotations

from enum import Enum

import pandas as pd

from fleetmix.benchmarking.converters.vrp import convert_vrp_to_fsm
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import Customer, FleetmixSolution, VehicleConfiguration
from fleetmix.optimization import solve_fsm_problem
from fleetmix.post_optimization import improve_solution
from fleetmix.preprocess.demand import maybe_explode
from fleetmix.utils.logging import log_detail, log_progress
from fleetmix.utils.time_measurement import TimeRecorder
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations


class VRPType(Enum):
    CVRP = "cvrp"
    MCVRP = "mcvrp"


def convert_to_fsm(vrp_type: VRPType, **kwargs) -> tuple[pd.DataFrame, Parameters]:
    """
    Library facade to convert VRP instances to FSM format.
    """
    return convert_vrp_to_fsm(vrp_type, **kwargs)


def run_optimization(
    customers_df: pd.DataFrame, params: Parameters, verbose: bool = False
) -> tuple[FleetmixSolution, list[VehicleConfiguration]]:
    """
    Run the common FSM optimization pipeline.
    Returns the solution object and the vehicleconfigurations list.
    """
    # Initialize TimeRecorder
    time_recorder = TimeRecorder()

    with time_recorder.measure("global"):
        # Generate vehicle configurations first (needed for smart explosion)
        with time_recorder.measure("vehicle_configuration"):
            configs = generate_vehicle_configurations(params.vehicles, params.goods)
            
        # Apply split-stop preprocessing if enabled
        allow_split = params.allow_split_stops
        customers_df = maybe_explode(customers_df, allow_split, configurations=configs)

        # Convert customers DataFrame to list of Customer objects
        customers = Customer.from_dataframe(customers_df)

        # Generate clusters
        with time_recorder.measure("clustering"):
            clusters = generate_clusters_for_configurations(
                customers=customers, configurations=configs, params=params
            )

        with time_recorder.measure("fsm_initial"):
            solution = solve_fsm_problem(
                clusters=clusters,
                configurations=configs,
                customers=customers,
                parameters=params,
                verbose=verbose,
                time_recorder=time_recorder,
            )

        # Apply post-optimization if enabled
        if params.post_optimization:
            with time_recorder.measure("fsm_post_optimization"):
                solution = improve_solution(solution, configs, customers, params)

    # Add time measurements to solution
    solution.time_measurements = time_recorder.measurements

    # Console output
    log_progress("Optimization Results:")
    log_detail(f"Total Cost: ${solution.total_cost:,.2f}")
    log_detail(f"Vehicles Used: {sum(solution.vehicles_used.values())}")
    log_detail(f"Expected Vehicles: {params.expected_vehicles}")

    return solution, configs
