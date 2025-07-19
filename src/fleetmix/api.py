"""
API facade for Fleetmix - provides a single entry point for programmatic usage.
"""

import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd

from fleetmix.clustering import generate_feasible_clusters
from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import Customer, FleetmixSolution, VehicleConfiguration
from fleetmix.optimization import optimize_fleet
from fleetmix.post_optimization import improve_solution
from fleetmix.preprocess.demand import maybe_explode
from fleetmix.utils.common import baseline_is_valid
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.logging import FleetmixLogger, log_warning
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.utils.time_measurement import TimeRecorder
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

logger = FleetmixLogger.get_logger("fleetmix.api")


def _two_phase_solve(
    customers_df: pd.DataFrame,
    configs: list[VehicleConfiguration],
    params: FleetmixParams,
    time_recorder: TimeRecorder,
    verbose: bool = False,
) -> FleetmixSolution:
    """Run two-phase optimization for split-stop mode."""
    # Phase 1: Baseline (no split stops)
    logger.info("Phase 1: Solving baseline problem without split stops")
    # Build a copy of params with allow_split_stops disabled
    baseline_problem = dataclasses.replace(params.problem, allow_split_stops=False)
    baseline_params = dataclasses.replace(params, problem=baseline_problem)

    # For phase 1, use the original customers without explosion
    baseline_customers_df = maybe_explode(
        customers_df, allow_split_stops=False, configurations=configs
    )
    baseline_customers = Customer.from_dataframe(baseline_customers_df)

    with time_recorder.measure("clustering_phase1"):
        baseline_clusters = generate_feasible_clusters(
            customers=baseline_customers, configurations=configs, params=baseline_params
        )

    with time_recorder.measure("fsm_phase1"):
        baseline_solution = optimize_fleet(
            clusters=baseline_clusters,
            configurations=configs,
            customers=baseline_customers,
            parameters=baseline_params,
            time_recorder=time_recorder,
        )

    # Apply post-optimization to phase 1 if enabled
    if params.algorithm.post_optimization:
        with time_recorder.measure("fsm_post_optimization_phase1"):
            baseline_solution = improve_solution(
                baseline_solution, configs, baseline_customers, baseline_params
            )

    baseline_vehicles = sum(baseline_solution.vehicles_used.values())
    logger.info(
        f"Phase 1 complete: {baseline_vehicles} vehicles, ${baseline_solution.total_cost:,.2f}"
    )

    # Check if baseline solution can be used as warm start
    baseline_valid = baseline_is_valid(baseline_solution)

    # Phase 2: Split-stop optimization with warm start
    logger.info("Phase 2: Solving split-stop problem with warm start")

    # Create a copy of parameters for phase 2
    phase2_problem = dataclasses.replace(params.problem, allow_split_stops=True)
    phase2_params = dataclasses.replace(params, problem=phase2_problem)

    # For phase 2, use the exploded customers
    phase2_customers_df = maybe_explode(
        customers_df, allow_split_stops=True, configurations=configs
    )
    phase2_customers = Customer.from_dataframe(phase2_customers_df)

    with time_recorder.measure("clustering_phase2"):
        phase2_clusters = generate_feasible_clusters(
            customers=phase2_customers, configurations=configs, params=phase2_params
        )

    try:
        with time_recorder.measure("fsm_phase2"):
            phase2_solution = optimize_fleet(
                clusters=phase2_clusters,
                configurations=configs,
                customers=phase2_customers,
                parameters=phase2_params,
                time_recorder=time_recorder,
                warm_start_solution=baseline_solution if baseline_valid else None,
            )

        # Apply post-optimization to phase 2 if enabled
        if params.algorithm.post_optimization:
            with time_recorder.measure("fsm_post_optimization_phase2"):
                phase2_solution = improve_solution(
                    phase2_solution, configs, phase2_customers, phase2_params
                )

        phase2_vehicles = sum(phase2_solution.vehicles_used.values())
        logger.info(
            f"Phase 2 complete: {phase2_vehicles} vehicles, ${phase2_solution.total_cost:,.2f}"
        )

        # Decide which solution to return
        if len(phase2_solution.missing_customers) == 0 and phase2_vehicles > 0:
            # Phase 2 feasible
            if not baseline_valid:
                logger.info("Baseline infeasible – using Phase 2 solution")
                return phase2_solution

            if (
                phase2_solution.total_cost < baseline_solution.total_cost
                and phase2_vehicles <= baseline_vehicles
            ):
                logger.info("Using Phase 2 solution (better cost and no more vehicles)")
                return phase2_solution

        logger.info("Using Phase 1 solution (Phase 2 not better or infeasible)")
        return baseline_solution

    except (ValueError, RuntimeError) as e:
        logger.warning(f"Phase 2 optimization failed: {e}, using baseline solution")
        return baseline_solution


# TODO: config solo string o path, no se puede pasar un objeto, despues simplificar
# handling de config
def optimize(
    demand: str | Path | pd.DataFrame,
    config: str | Path | FleetmixParams | None = None,
    output_dir: str = "results",
    format: str = "json",
    verbose: bool = False,
    allow_split_stops: Optional[bool] = None,
) -> FleetmixSolution:
    """
    Optimize fleet size and mix for given demand and configuration.

    Args:
        demand: Customer demand data - can be:
            - Path to CSV/Excel file containing customer demand data
            - Pandas DataFrame with demand data
        config: Configuration parameters - can be:
            - Path to YAML configuration file
            - Parameters object
            - None (uses default configuration)
        output_dir: Directory to save results (default: "results")
        format: Output format - "xlsx" or "json" (default: "json")
        verbose: Enable verbose logging (default: False)
        allow_split_stops: Allow customers to be served by multiple vehicles (default: False)

    Returns:
        FleetmixSolution: Optimization results

    Raises:
        FileNotFoundError: If demand file or config file doesn't exist
        ValueError: If demand data is invalid or optimization fails

    Example:
        >>> # Optimize using file paths
        >>> solution = optimize("demand.csv", "config.yaml")
        >>> print(f"Total cost: ${solution.total_cost:,.2f}")
        >>> print(f"Vehicles used: {solution.total_vehicles}")

        >>> # Optimize using DataFrame and FleetmixParams object
        >>> import pandas as pd
        >>> demand_df = pd.read_csv("demand.csv")
        >>> params = load_fleetmix_params("config.yaml")
        >>> solution = optimize(demand_df, params, verbose=True)
    """

    # Initialize TimeRecorder
    time_recorder = TimeRecorder()

    with time_recorder.measure("global"):
        # Step 1: Load demand data
        if isinstance(demand, pd.DataFrame):
            customers_df = demand.copy()
            logger.info("Using provided DataFrame for customer demand data")
        else:
            demand_path = Path(demand)
            if not demand_path.exists():
                raise FileNotFoundError(
                    f"Demand file not found: {demand_path}\n"
                    f"Please check the file path and ensure it exists."
                )
            try:
                with time_recorder.measure("load_demand"):
                    # Try to read the CSV directly first
                    df = pd.read_csv(demand_path)

                    # Check if it's already in wide format (has demand columns)
                    if any(col.endswith("_Demand") for col in df.columns):
                        # Already in wide format, use it directly
                        customers_df = df
                    else:
                        # It's in long format, process it
                        customers_df = load_customer_demand(str(demand_path))
                logger.info(f"Loaded {len(customers_df)} customers from {demand_path}")
            except Exception as e:
                raise ValueError(
                    f"Error loading demand data from {demand_path}:\n{e!s}\n"
                    f"Please check the file format and ensure it contains valid demand data."
                )

        # Validate demand data
        if customers_df.empty:
            raise ValueError(
                "Demand data is empty. Please provide a file with customer demand data."
            )

        # Step 2: Load parameters
        if config is None:
            # Search default locations
            default_paths = [
                Path.cwd() / "config.yaml",
                Path(__file__).parent / "config" / "default_config.yaml",
                Path(__file__).parent / "config" / "baseline_config.yaml",
            ]
            for p in default_paths:
                if p.exists():
                    params = load_fleetmix_params(p)
                    break
            else:
                raise FileNotFoundError(
                    "No configuration file provided and no default config found."
                )
        elif isinstance(config, FleetmixParams):
            params = config
        else:
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}\n"
                    f"Please check the file path and ensure it exists."
                )
            try:
                params = load_fleetmix_params(config_path)
            except Exception as e:
                raise ValueError(
                    f"Error loading configuration from {config_path}:\n{e!s}\n"
                    f"Please check the YAML syntax and required fields."
                )

        # Override allow_split_stops if provided via API
        if allow_split_stops is not None:
            params = dataclasses.replace(
                params,
                problem=dataclasses.replace(
                    params.problem, allow_split_stops=allow_split_stops
                ),
            )

        # Update demand_file and results_dir in params to reflect the actual values being used
        if isinstance(demand, (str, Path)):
            params = dataclasses.replace(
                params, io=dataclasses.replace(params.io, demand_file=str(demand))
            )

        # Update results_dir if output_dir is provided
        if output_dir:
            params = dataclasses.replace(
                params, io=dataclasses.replace(params.io, results_dir=Path(output_dir))
            )

        if verbose:
            params = dataclasses.replace(
                params, runtime=dataclasses.replace(params.runtime, verbose=True)
            )

        # Validate demand DataFrame has required columns
        required_columns = ["Customer_ID", "Latitude", "Longitude"]
        demand_columns = [f"{good}_Demand" for good in params.problem.goods]
        required_columns.extend(demand_columns)

        missing_columns = [
            col for col in required_columns if col not in customers_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Required columns are: {required_columns}\n"
                f"Available columns are: {list(customers_df.columns)}"
            )

        # Step 3: Generate vehicle configurations
        try:
            with time_recorder.measure("vehicle_configuration"):
                configs = generate_vehicle_configurations(
                    params.problem.vehicles, params.problem.goods
                )
        except Exception as e:
            raise ValueError(
                f"Error generating vehicle configurations:\n{e!s}\n"
                f"Please check your vehicle and goods definitions in the config."
            )

        # Step 4: Solve optimization problem
        # Use two-phase approach if split stops are enabled
        if params.problem.allow_split_stops:
            solution = _two_phase_solve(
                customers_df=customers_df,
                configs=configs,
                params=params,
                time_recorder=time_recorder,
                verbose=verbose,
            )
        else:
            # Standard single-phase optimization
            # Apply split-stop preprocessing if needed
            customers_df = maybe_explode(
                customers_df, params.problem.allow_split_stops, configurations=configs
            )
            customers = Customer.from_dataframe(customers_df)

            # Step 4a: Generate clusters
            try:
                with time_recorder.measure("clustering"):
                    clusters = generate_feasible_clusters(
                        customers=customers, configurations=configs, params=params
                    )
            except Exception as e:
                raise ValueError(
                    f"Error generating customer clusters:\n{e!s}\n"
                    f"This could be due to incompatible vehicle capacities, "
                    f"time constraints, or compartment configurations."
                )

            # Check if clustering generated any valid clusters
            if not clusters:
                raise ValueError(
                    "No feasible clusters could be generated!\n"
                    "Possible causes:\n"
                    "- Vehicle capacities are too small for customer demands\n"
                    "- No vehicles have the right compartment configuration for customer demands\n"
                    "- Time constraints are too restrictive\n"
                    "Please review your configuration and customer data."
                )

            # Step 4b: Solve optimization
            try:
                with time_recorder.measure("fsm_initial"):
                    solution = optimize_fleet(
                        clusters=clusters,
                        configurations=configs,
                        customers=customers,
                        parameters=params,
                        time_recorder=time_recorder,
                    )

                # Step 5: Post-optimization improvement if enabled
                if params.algorithm.post_optimization:
                    with time_recorder.measure("fsm_post_optimization"):
                        solution = improve_solution(
                            solution, configs, customers, params
                        )
            except Exception as e:
                raise ValueError(
                    f"Error during optimization:\n{e!s}\n"
                    f"This could be due to infeasible problem constraints "
                    f"or insufficient cluster coverage."
                )

    # Add time measurements to solution
    solution.time_measurements = time_recorder.measurements

    # Step 6: Save results if output directory is specified
    if output_dir:
        try:
            save_optimization_results(
                solution=solution,
                parameters=params,
                format=format,
            )
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            log_warning(f"Failed to save results: {e!s}")

    return solution
