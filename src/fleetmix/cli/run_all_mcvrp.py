"""Batch runner for all MCVRP instances: convert to FSM, optimize, and save JSON results."""
import time
from pathlib import Path

import sys

from fleetmix.utils.logging import setup_logging
from fleetmix.benchmarking.converters.mcvrp import convert_mcvrp_to_fsm
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.cli.main import solve_fsm_problem
from fleetmix.utils.save_results import save_optimization_results

def main():
    setup_logging()
    inst_dir = Path(__file__).parent.parent / "benchmarking" / "datasets" / "mcvrp"
    # results_dir will be determined by the params object, initialized per instance
    # The directory params.results_dir should already be created by Parameters.__post_init__

    for dat_path in sorted(inst_dir.glob("*.dat")):
        instance = dat_path.stem
        print(f"Running instance {instance}...")
        customers_df, params = convert_mcvrp_to_fsm(dat_path)
        configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
        clusters_df = generate_clusters_for_configurations(
            customers=customers_df,
            configurations_df=configs_df,
            params=params
        )
        start_time = time.time()
        solution = solve_fsm_problem(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=customers_df,
            parameters=params,
            verbose=False
        )

        output_path = params.results_dir / f"mcvrp_{instance}.json"
        save_optimization_results(
            execution_time=time.time() - start_time,
            solver_name=solution["solver_name"],
            solver_status=solution["solver_status"],
            solver_runtime_sec=solution["solver_runtime_sec"],
            post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
            configurations_df=configs_df,
            selected_clusters=solution["selected_clusters"],
            total_fixed_cost=solution["total_fixed_cost"],
            total_variable_cost=solution["total_variable_cost"],
            total_light_load_penalties=solution["total_light_load_penalties"],
            total_compartment_penalties=solution["total_compartment_penalties"],
            total_penalties=solution["total_penalties"],
            vehicles_used=solution["vehicles_used"],
            missing_customers=solution["missing_customers"],
            parameters=params,
            filename=str(output_path),
            format="json",
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles
        )
        print(f"Saved results to {output_path.name}")

if __name__ == "__main__":
    main() 