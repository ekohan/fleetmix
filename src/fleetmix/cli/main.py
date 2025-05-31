"""
Main module for the vehicle routing optimization problem.
"""
import time
from pathlib import Path

from fleetmix.utils.cli import parse_args, load_parameters, print_parameter_help
from fleetmix.utils.logging import (
    setup_logging, ProgressTracker, LogLevel, log_progress, log_success, log_detail
)
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.optimization import solve_fsm_problem
from fleetmix.utils.time_measurement import TimeRecorder

def main():
    """Run the FSM optimization pipeline."""
    # Add deprecation warning
    from warnings import warn
    warn("Direct script execution is deprecated. Use 'fleetmix optimize' instead", FutureWarning)
    
    # Parse arguments and load parameters
    parser = parse_args()
    args = parser.parse_args()
    
    # Check for help flag and exit if requested
    if args.help_params:
        print_parameter_help()
        return
    
    # Setup logging with appropriate level
    if hasattr(args, 'debug') and args.debug:
        setup_logging(LogLevel.DEBUG)
    elif args.verbose:
        setup_logging(LogLevel.VERBOSE)
    elif hasattr(args, 'quiet') and args.quiet:
        setup_logging(LogLevel.QUIET)
    else:
        setup_logging(LogLevel.NORMAL)
        
    params = load_parameters(args)
    
    # Initialize TimeRecorder
    time_recorder = TimeRecorder()
    
    # Define optimization steps
    steps = [
        'Load Data',
        'Generate Configs',
        'Create Clusters',
        'Optimize Fleet',
        'Save Results'
    ]
    
    progress = ProgressTracker(steps)
    
    # Start global time measurement
    with time_recorder.measure("global"):
        start_time = time.time()

        # Step 1: Load customer data
        customers = load_customer_demand(params.demand_file)
        progress.advance(f"Loaded {len(customers)} customers")

        # Step 2: Generate vehicle configurations
        with time_recorder.measure("vehicle_configuration"):
            configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
        progress.advance(f"Generated {len(configs_df)} vehicle configurations")

        # Step 3: Generate clusters
        with time_recorder.measure("clustering"):
            clusters_df = generate_clusters_for_configurations(
                customers=customers,
                configurations_df=configs_df,
                params=params
            )
        progress.advance(f"Created {len(clusters_df)} clusters")

        # Step 4: Solve optimization problem
        with time_recorder.measure("fsm_initial"):
            solution = solve_fsm_problem(
                clusters_df=clusters_df,
                configurations_df=configs_df,
                customers_df=customers,
                parameters=params,
                verbose=args.verbose,
                time_recorder=time_recorder
            )
        total_cost = solution.total_fixed_cost + solution.total_variable_cost + solution.total_penalties
        progress.advance(f"Optimized fleet: ${total_cost:,.2f} total cost")

        solution.time_measurements = time_recorder.measurements

        # Step 5: Save results
        save_optimization_results(
            solution=solution,
            configurations_df=configs_df,
            parameters=params,
            format=args.format
        )
        progress.advance(f"Results saved (execution time: {time.time() - start_time:.1f}s)")
    
    progress.close()

if __name__ == "__main__":
    main() 