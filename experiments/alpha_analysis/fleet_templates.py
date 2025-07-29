"""
Fleet template builders for alpha analysis.

Returns FleetmixParams instances for SCV and MCV fleets.
"""

from fleetmix.config.params import FleetmixParams, ProblemParams, VehicleSpec
from fleetmix.config import load_fleetmix_params
from pathlib import Path
import dataclasses

BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config.yaml")

def make_scv_fleet(demand_day: str) -> FleetmixParams:
    """Create params for SCV-only fleet (one vehicle type per good)."""
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    goods = base_params.problem.goods
    base_vehicle = next(iter(base_params.problem.vehicles.values()))  # Use first as template
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