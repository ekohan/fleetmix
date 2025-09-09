"""
Fleet template builders for alpha analysis.

Returns FleetmixParams instances for SCV and MCV fleets.
"""

import dataclasses
from pathlib import Path

from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams, ProblemParams
from fleetmix.core_types import VehicleSpec
from fleetmix.utils.logging import FleetmixLogger

BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config_experiments.yaml")
logger = FleetmixLogger.get_logger(__name__)


def make_scv_fleet(demand_day: str) -> FleetmixParams:
    """Create params for SCV-only fleet (one vehicle type per good)."""
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    goods = base_params.problem.goods
    base_vehicle = next(
        iter(base_params.problem.vehicles.values())
    )  # Use first as template
    scv_vehicles = {}
    for good in goods:
        scv_spec = dataclasses.replace(
            base_vehicle,
            allowed_goods=[good],
        )
        scv_vehicles[f"SCV_{good}"] = scv_spec
    scv_problem = dataclasses.replace(
        base_params.problem,
        vehicles=scv_vehicles,
        compartment_setup_cost=0.0,  # No extra compartments for SCV
        allow_split_stops=True,  # Required for SCVs to serve multi-good customers
    )
    io_params = dataclasses.replace(base_params.io, demand_file=demand_day)
    return dataclasses.replace(base_params, problem=scv_problem, io=io_params)


def make_mcv_fleet(alpha: float, C: float, demand_day: str) -> FleetmixParams:
    """Create params for MCV fleet with alpha multiplier and C setup cost."""
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    mcv_vehicles = {}
    for vt, spec in base_params.problem.vehicles.items():
        mcv_spec = dataclasses.replace(
            spec,
            fixed_cost=spec.fixed_cost * alpha,
        )
        mcv_vehicles[vt] = mcv_spec
    mcv_problem = dataclasses.replace(
        base_params.problem,
        vehicles=mcv_vehicles,
        compartment_setup_cost=C,
    )
    io_params = dataclasses.replace(base_params.io, demand_file=demand_day)
    return dataclasses.replace(base_params, problem=mcv_problem, io=io_params)


def make_mixed_fleet(
    alpha: float, C: float, demand_day: str, allow_split: bool | None = None
) -> FleetmixParams:
    """Create params for mixed fleet (SCV + MCV) with alpha multiplier on MCV and C setup cost."""
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    goods = base_params.problem.goods
    base_vehicle = next(
        iter(base_params.problem.vehicles.values())
    )  # Use first as template

    # SCV vehicles: one per good
    scv_vehicles: dict[str, VehicleSpec] = {}
    for good in goods:
        scv_spec = dataclasses.replace(base_vehicle, allowed_goods=[good])
        scv_vehicles[f"SCV_{good}"] = scv_spec

    # MCV vehicles: use all defined vehicles, apply alpha multiplier
    mcv_vehicles: dict[str, VehicleSpec] = {}
    for vt, spec in base_params.problem.vehicles.items():
        mcv_spec = dataclasses.replace(spec, fixed_cost=spec.fixed_cost * alpha)
        mcv_vehicles[f"MCV_{vt}"] = mcv_spec  # prefix to avoid collisions

    # Combine both vehicle types
    mixed_vehicles = {**scv_vehicles, **mcv_vehicles}

    problem = dataclasses.replace(
        base_params.problem,
        vehicles=mixed_vehicles,
        compartment_setup_cost=C,
        allow_split_stops=allow_split
        if allow_split is not None
        else base_params.problem.allow_split_stops,
    )
    io_params = dataclasses.replace(base_params.io, demand_file=demand_day)
    params = dataclasses.replace(base_params, problem=problem, io=io_params)
    # Light debug: confirm presence of SCV and MCV vehicle types
    try:
        scv_names = [k for k in mixed_vehicles if k.startswith("SCV_")]
        mcv_names = [k for k in mixed_vehicles if k.startswith("MCV_")]
        logger.debug(
            f"Mixed fleet built → SCV={len(scv_names)} types, MCV={len(mcv_names)} types; C={C}, alpha={alpha}"
        )
    except Exception:
        pass
    return params
