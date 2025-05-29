"""
Batch runner for all CVRP instances: convert to FSM, optimize, and save JSON results.

This module is a thin wrapper around the unified VRP-to-FSM pipeline interface (vrp_to_fsm.py).
It processes all CVRP instances in the datasets/cvrp directory using benchmark-type=normal mode,
applying the same conversion and optimization logic but in batch mode with JSON output.

"""
import time
from pathlib import Path
from warnings import warn

from fleetmix.utils.logging import setup_logging
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType
from fleetmix.utils.save_results import save_optimization_results

def main():
    """Run benchmarks for all CVRP instances."""
    # Add deprecation warning
    warn("Direct script execution is deprecated. Use 'fleetmix benchmark cvrp' instead", FutureWarning)
    
    setup_logging()
    inst_dir = Path(__file__).parent.parent / "benchmarking" / "datasets" / "cvrp"

    for vrp_path in sorted(inst_dir.glob("X-n*.vrp")):
        instance = vrp_path.stem
        print(f"Running CVRP instance {instance}...")
        
        # Use the unified pipeline interface for conversion
        # CVRP requires benchmark_type and uses instance_names instead of instance_path
        customers_df, params = convert_to_fsm(
            VRPType.CVRP,
            instance_names=[instance],
            benchmark_type=CVRPBenchmarkType.NORMAL
        )
        
        # Use the unified pipeline interface for optimization
        start_time = time.time()
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )

        # Save results as JSON for later batch analysis
        output_path = params.results_dir / f"cvrp_{instance}_normal.json"
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