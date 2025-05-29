"""
Batch runner for all MCVRP instances: convert to FSM, optimize, and save JSON results.

This module is a thin wrapper around the unified VRP-to-FSM pipeline interface (vrp_to_fsm.py).
It processes all MCVRP instances in the datasets/mcvrp directory, applying the same conversion
and optimization logic but in batch mode with JSON output.

"""
import time
from pathlib import Path
from warnings import warn

from fleetmix.utils.logging import setup_logging
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.utils.save_results import save_optimization_results

def main():
    """Run benchmarks for all MCVRP instances."""
    # Add deprecation warning
    warn("Direct script execution is deprecated. Use 'fleetmix benchmark mcvrp' instead", FutureWarning)
    
    setup_logging()
    inst_dir = Path(__file__).parent.parent / "benchmarking" / "datasets" / "mcvrp"

    for dat_path in sorted(inst_dir.glob("*.dat")):
        instance = dat_path.stem
        print(f"Running instance {instance}...")
        
        # Use the unified pipeline interface for conversion
        customers_df, params = convert_to_fsm(
            VRPType.MCVRP,
            instance_path=dat_path
        )
        
        # Use the unified pipeline interface for optimization
        start_time = time.time()
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )

        # Save results as JSON for later batch analysis
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