"""
API facade for Fleetmix - provides a single entry point for programmatic usage.
"""

from pathlib import Path

import pandas as pd

from fleetmix.clustering import generate_feasible_clusters
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import Customer, FleetmixSolution, VehicleConfiguration
from fleetmix.optimization import optimize_fleet
from fleetmix.preprocess.demand import maybe_explode
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.logging import FleetmixLogger, log_warning
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.utils.time_measurement import TimeRecorder
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

logger = FleetmixLogger.get_logger("fleetmix.api")


def optimize(
    demand: str | Path | pd.DataFrame,
    config: str | Path | Parameters | None = None,
    output_dir: str = "results",
    format: str = "excel",
    verbose: bool = False,
    allow_split_stops: bool = False,
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
        format: Output format - "excel" or "json" (default: "excel")
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

        >>> # Optimize using DataFrame and Parameters object
        >>> import pandas as pd
        >>> demand_df = pd.read_csv("demand.csv")
        >>> params = Parameters.from_yaml("config.yaml")
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
                    f"Error loading configuration from {config_path}:\n{e!s}\n"
                    f"Please check the YAML syntax and required fields."
                )

        # Override allow_split_stops if provided via API
        if allow_split_stops != params.allow_split_stops:
            params.allow_split_stops = allow_split_stops

        # Apply split-stop preprocessing if enabled
        customers_df = maybe_explode(customers_df, params.allow_split_stops)

        # Validate demand DataFrame has required columns
        required_columns = ["Customer_ID", "Latitude", "Longitude"]
        demand_columns = [f"{good}_Demand" for good in params.goods]
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

        # Convert customers DataFrame to list of Customer objects
        customers = Customer.from_dataframe(customers_df)

        # Step 3: Generate vehicle configurations
        try:
            with time_recorder.measure("vehicle_configuration"):
                configs = generate_vehicle_configurations(params.vehicles, params.goods)
        except Exception as e:
            raise ValueError(
                f"Error generating vehicle configurations:\n{e!s}\n"
                f"Please check your vehicle and goods definitions in the config."
            )

        # Step 4: Generate clusters
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

        # Step 5: Solve optimization problem
        try:
            with time_recorder.measure("fsm_initial"):
                solution = optimize_fleet(
                    clusters=clusters,
                    configurations=configs,
                    customers=customers,
                    parameters=params,
                    verbose=verbose,
                    time_recorder=time_recorder,
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
                configurations=configs,
                parameters=params,
                format=format,
            )
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            log_warning(f"Failed to save results: {e!s}")

    return solution
