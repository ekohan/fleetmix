import dataclasses
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Any

from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.utils.logging import Colors


def print_parameter_help():
    """Display detailed help information about parameters"""
    help_text = f"""
{Colors.BOLD}Fleet Size and Mix Optimization Parameters{Colors.RESET}
{Colors.CYAN}════════════════════════════════════════{Colors.RESET}

{Colors.YELLOW}Core Parameters:{Colors.RESET}
  --route-time-estimation STR
                           Method to estimate route times
                           Options: 
                             - Legacy (simple service time based)
                             - BHH (Beardwood-Halton-Hammersley)
                             - TSP (exact TSP solver)
                           Default: BHH
                           Example: --route-time-estimation BHH

  --light-load-penalty FLOAT
                           Penalty cost for light loads
                           Set to 0 to disable penalties
                           Default: 1000
                           Example: --light-load-penalty 500

  --light-load-threshold FLOAT
                           Threshold for light load penalty (0.0 to 1.0)
                           Example: 0.2 means penalize loads below 20%
                           Default: 0.20
                           Example: --light-load-threshold 0.3

  --compartment-setup-cost FLOAT
                           Cost per additional compartment beyond the first one
                           Example: 50 means $50 extra for second compartment, $100 for third
                           Default: 50
                           Example: --compartment-setup-cost 75

{Colors.YELLOW}Clustering Options:{Colors.RESET}
  --clustering-method STR  Method to cluster customers
                           Options:
                             - minibatch_kmeans (fast k-means variant)
                             - kmedoids (k-medoids clustering)
                             - agglomerative (hierarchical clustering)
                             - combine (use all methods in tandem for each config)
                           Default: minibatch_kmeans
                           Example: --clustering-method combine

  --clustering-distance STR
                           Distance metric for clustering
                           Options:
                             - euclidean (straight-line distance)
                             - composite (geo + demand, agglomerative only)
                           Default: euclidean
                           Example: --clustering-distance composite

  --geo-weight FLOAT      Weight for geographical distance (0.0 to 1.0)
                           Default: 0.7
                           Example: --geo-weight 0.8

  --demand-weight FLOAT   Weight for demand distance (0.0 to 1.0)
                           Default: 0.3
                           Example: --demand-weight 0.2

{Colors.YELLOW}Input/Output:{Colors.RESET}
  --demand-file STR        Name of the demand file to use
                           Must be in the data directory
                           Default: Defined in config file
                           Example: --demand-file sales_2023_high_demand_day.csv

  --config PATH            Path to custom config file
                           Default: src/config/default_config.yaml
                           Example: --config my_config.yaml

  --format STR            Output format (xlsx, json)
                           Default: json
                           Example: --format xlsx

{Colors.YELLOW}Other Options:{Colors.RESET}
  --verbose               Enable verbose output
                           Default: False
                           Example: --verbose

{Colors.CYAN}Examples:{Colors.RESET}
  # Use custom config file
  python src/main.py --config my_config.yaml

  # Override specific parameters
  python src/main.py --light-load-penalty 500 --compartment-setup-cost 75

  # Change clustering method and distance metric
  python src/main.py --clustering-method agglomerative --clustering-distance composite

  # Use different demand file with verbose output
  python src/main.py --demand-file sales_2023_high_demand_day.csv --verbose

{Colors.YELLOW}Note:{Colors.RESET}
  Vehicle-specific parameters (avg_speed, service_time, max_route_time) are now
  configured per vehicle type in the config file and cannot be overridden globally.
"""
    print(help_text)
    sys.exit(0)


def parse_args() -> ArgumentParser:
    """Parse command line arguments for parameter overrides"""
    parser = ArgumentParser(
        description="Fleet Size and Mix Optimization",
        formatter_class=RawTextHelpFormatter,
    )

    # Add help-params argument
    parser.add_argument(
        "--help-params",
        action="store_true",
        help="Show detailed parameter information and exit",
    )

    # Add arguments for each parameter that can be overridden
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument(
        "--demand-file", type=str, help="Name of the demand file to use"
    )
    parser.add_argument(
        "--light-load-penalty", type=float, help="Penalty for light loads"
    )
    parser.add_argument(
        "--light-load-threshold", type=float, help="Threshold for light load penalty"
    )
    parser.add_argument(
        "--compartment-setup-cost",
        type=float,
        help="Cost per additional compartment beyond the first one",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--route-time-estimation",
        type=str,
        choices=["BHH", "TSP", "Legacy"],
        help="Method to estimate route times (BHH, TSP, Legacy)",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=["minibatch_kmeans", "kmedoids", "agglomerative", "combine"],
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--clustering-distance",
        type=str,
        choices=["euclidean", "composite"],
        help="Distance metric for clustering (composite only for agglomerative)",
    )
    parser.add_argument(
        "--geo-weight", type=float, help="Weight for geographical distance (0.0 to 1.0)"
    )
    parser.add_argument(
        "--demand-weight", type=float, help="Weight for demand distance (0.0 to 1.0)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xlsx", "json"],
        help="Output format (xlsx, json)",
        default="json",
    )

    return parser


def get_parameter_overrides(args) -> dict[str, Any]:
    """Extract parameter overrides from command line arguments"""
    # Convert args to dictionary, excluding None values
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    # Remove non-parameter arguments
    for key in ["config", "verbose", "help_params"]:
        overrides.pop(key, None)

    # Remove old timing parameters that are now vehicle-specific
    for key in ["avg_speed", "service_time", "max_route_time"]:
        overrides.pop(key, None)

    # Convert dashed args to underscores
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}

    return overrides


def load_parameters(args) -> FleetmixParams:
    """Load parameters with optional command line overrides"""
    # Load base parameters
    if args.config:
        params = load_fleetmix_params(args.config)
    else:
        # Search for default config in standard locations
        from pathlib import Path

        default_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config" / "default_config.yaml",
            Path(__file__).parent.parent / "config" / "baseline_config.yaml",
        ]
        for p in default_paths:
            if p.exists():
                params = load_fleetmix_params(p)
                break
        else:
            raise FileNotFoundError(
                "No configuration file provided and no default config found."
            )

    # Get overrides from command line
    overrides = get_parameter_overrides(args)

    if not overrides:
        return params

    # Handle clustering parameters by updating algorithm section
    clustering_override_keys = [
        "clustering_method",
        "clustering_distance",
        "geo_weight",
        "demand_weight",
        "route_time_estimation",
    ]

    # TODO ver si conservar o no los overrides
    algorithm_updates = {}

    if "clustering_method" in overrides:
        algorithm_updates["clustering_method"] = overrides.pop("clustering_method")
    if "clustering_distance" in overrides:
        algorithm_updates["clustering_distance"] = overrides.pop("clustering_distance")
    if "geo_weight" in overrides:
        algorithm_updates["geo_weight"] = overrides.pop("geo_weight")
    if "demand_weight" in overrides:
        algorithm_updates["demand_weight"] = overrides.pop("demand_weight")
    if "route_time_estimation" in overrides:
        algorithm_updates["route_time_estimation"] = overrides.pop(
            "route_time_estimation"
        )

    # Handle problem-level parameters
    problem_updates = {}
    if "light_load_penalty" in overrides:
        problem_updates["light_load_penalty"] = overrides.pop("light_load_penalty")
    if "light_load_threshold" in overrides:
        problem_updates["light_load_threshold"] = overrides.pop("light_load_threshold")
    if "compartment_setup_cost" in overrides:
        problem_updates["compartment_setup_cost"] = overrides.pop(
            "compartment_setup_cost"
        )

    # Handle IO parameters
    io_updates = {}
    if "demand_file" in overrides:
        io_updates["demand_file"] = overrides.pop("demand_file")
    if "format" in overrides:
        io_updates["format"] = overrides.pop("format")

    # Apply updates using dataclasses.replace
    if problem_updates:
        params = dataclasses.replace(
            params, problem=dataclasses.replace(params.problem, **problem_updates)
        )

    if algorithm_updates:
        params = dataclasses.replace(
            params, algorithm=dataclasses.replace(params.algorithm, **algorithm_updates)
        )

    if io_updates:
        params = dataclasses.replace(
            params, io=dataclasses.replace(params.io, **io_updates)
        )

    # Handle any remaining overrides (should be empty now, but just in case)
    if overrides:
        raise ValueError(f"Unknown parameter overrides: {list(overrides.keys())}")

    return params
