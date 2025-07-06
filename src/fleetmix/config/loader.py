from __future__ import annotations

"""Utilities for loading FleetMix configuration YAML files into the
parameter dataclass hierarchy.

The YAML structure expected by the public docs remains *unchanged* so that all
existing configuration files continue to function, while the internal data
representation becomes more structured and type-safe.
"""

from pathlib import Path
from typing import Dict, Any

import yaml

from fleetmix.core_types import VehicleSpec, DepotLocation
from fleetmix.utils.logging import FleetmixLogger

from .params import ProblemParams, AlgorithmParams, IOParams, FleetmixParams

logger = FleetmixLogger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helper parsing routines
# ---------------------------------------------------------------------------


def _parse_vehicles(raw: Dict[str, Dict[str, Any]]) -> Dict[str, VehicleSpec]:
    """Convert YAML mapping of vehicle specs into `VehicleSpec` instances."""

    parsed: Dict[str, VehicleSpec] = {}
    for name, details in raw.items():
        # Mandatory fields
        spec_kwargs: Dict[str, Any] = {
            "capacity": details.pop("capacity"),
            "fixed_cost": details.pop("fixed_cost"),
            "avg_speed": details.pop("avg_speed"),
            "service_time": details.pop("service_time"),
            "max_route_time": details.pop("max_route_time"),
        }

        # Optional allowed_goods
        if "allowed_goods" in details:
            spec_kwargs["allowed_goods"] = details.pop("allowed_goods")

        # Remaining extra fields
        spec_kwargs["extra"] = details
        parsed[name] = VehicleSpec(**spec_kwargs)

    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> FleetmixParams:
    """Load YAML configuration file into `FleetmixParams`.

    The function adheres to the *current* flat/nested YAML schema so that users
    need not update any existing files.
    """

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    try:
        with cfg_path.open() as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Error parsing YAML configuration {cfg_path}: {exc}") from exc

    # ---------------------------------------------------------------------
    # Extract problem definition
    # ---------------------------------------------------------------------

    try:
        vehicles_raw = data.pop("vehicles")
    except KeyError as exc:
        raise ValueError("YAML missing required key 'vehicles'.") from exc

    vehicles = _parse_vehicles(vehicles_raw)

    try:
        depot_raw = data.pop("depot")
    except KeyError as exc:
        raise ValueError("YAML missing required key 'depot'.") from exc

    depot = DepotLocation(**depot_raw)

    goods = data.pop("goods", [])
    variable_cph = data.pop("variable_cost_per_hour", 0.0)
    light_load_penalty = data.pop("light_load_penalty", 0.0)
    light_load_threshold = data.pop("light_load_threshold", 0.0)
    compartment_setup_cost = data.pop("compartment_setup_cost", 0.0)
    allow_split_stops = data.pop("allow_split_stops", False)

    problem = ProblemParams(
        vehicles=vehicles,
        depot=depot,
        goods=goods,
        variable_cost_per_hour=variable_cph,
        light_load_penalty=light_load_penalty,
        light_load_threshold=light_load_threshold,
        compartment_setup_cost=compartment_setup_cost,
        allow_split_stops=allow_split_stops,
    )

    # ---------------------------------------------------------------------
    # Algorithm parameters
    # ---------------------------------------------------------------------

    clustering = data.pop("clustering", {}) or {}

    algorithm = AlgorithmParams(
        clustering_max_depth=clustering.get("max_depth", 20),
        clustering_method=clustering.get("method", "combine"),
        clustering_distance=clustering.get("distance", "euclidean"),
        geo_weight=clustering.get("geo_weight", 0.7),
        demand_weight=clustering.get("demand_weight", 0.3),
        route_time_estimation=clustering.get("route_time_estimation", "BHH"),
        prune_tsp=data.pop("prune_tsp", False),
        small_cluster_size=data.pop("small_cluster_size", 1000),
        nearest_merge_candidates=data.pop("nearest_merge_candidates", 1000),
        max_improvement_iterations=data.pop("max_improvement_iterations", 20),
        pre_small_cluster_size=data.pop("pre_small_cluster_size", 5),
        pre_nearest_merge_candidates=data.pop("pre_nearest_merge_candidates", 50),
        post_optimization=data.pop("post_optimization", True),
    )

    # ---------------------------------------------------------------------
    # IO parameters
    # ---------------------------------------------------------------------

    demand_file = data.pop("demand_file", None)
    if demand_file is None:
        raise ValueError("YAML missing required key 'demand_file'.")

    results_dir_path = Path(data.pop("results_dir", "results"))
    fmt = data.pop("format", "json")

    io_params = IOParams(
        demand_file=demand_file,
        results_dir=results_dir_path,
        format=fmt,
    )

    # Any remaining unknown keys will raise an error to avoid silent mistakes.
    if data:
        unknown_keys = ", ".join(sorted(data.keys()))
        raise ValueError(
            f"Unknown top-level configuration keys in YAML: {unknown_keys}"
        )

    logger.debug(
        "Loaded configuration – problem: %s algorithm: %s io: %s",
        problem,
        algorithm,
        io_params,
    )

    return FleetmixParams(problem=problem, algorithm=algorithm, io=io_params) 