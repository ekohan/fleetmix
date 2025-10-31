"""
Day-level optimization runner.
"""

from pathlib import Path
from typing import Any

import numpy as np

from fleetmix.api import optimize

# For derived parameter expressions
from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.experiments.alpha_analysis.metrics import (
    average_visits_per_customer,
    cost_per_drop,
    cost_per_kg,
    distance_ratios,
    route_time_stats,
    split_rate,
    stops_stats,
)
from fleetmix.utils.data_processing import load_customer_demand

BASE_CONFIG_PATH = Path(
    "src/fleetmix/config/experiments/fleet_composition/base_config.yaml"
)


# Copied utility from run_grid to safely convert numpy & other objects to native types
def _convert_numpy_types(obj) -> Any:  # local func to keep this module self-contained
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [_convert_numpy_types(x) for x in obj]
        return conv if isinstance(obj, list) else tuple(conv)
    return obj


def run_day(
    demand_path: Path,
    fleet_params: FleetmixParams,
    fleet_type: str,
    alpha: float = 1.0,
    C: float = 0.0,
) -> dict[Any, Any]:
    """Run optimization for one demand day and compute metrics."""
    customers_df = load_customer_demand(str(demand_path))
    num_customers = len(customers_df)
    total_demand_kg = (
        customers_df[[col for col in customers_df.columns if col.endswith("_Demand")]]
        .sum()
        .sum()
    )
    solution = optimize(customers_df, fleet_params)
    solver_cost = solution.total_cost
    cost_per_d = cost_per_drop(solver_cost, num_customers)
    cost_per_k = cost_per_kg(solver_cost, total_demand_kg)
    split_r = split_rate(solution)
    avg_visits = average_visits_per_customer(solution)
    # Base SCV fixed cost for percentage calculations
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    base_fixed_cost = float(
        next(iter(base_params.problem.vehicles.values())).fixed_cost
    )

    alpha_surcharge_pct = 100 * (alpha - 1)
    c_pct_scv = 100 * (C / base_fixed_cost) if base_fixed_cost else 0.0

    # Add detailed metrics from solution
    total_route_time_hours = (
        solution.total_variable_cost / fleet_params.problem.variable_cost_per_hour
        if fleet_params.problem.variable_cost_per_hour > 0
        else 0.0
    )

    # Route-level stats
    rt_stats = route_time_stats(solution)
    st_stats = stops_stats(solution)

    # Estimate total distance similar to run_grid_mixed logic (route_time * avg_speed)
    total_distance_km = 0.0
    for cl in solution.selected_clusters:
        vt_spec = fleet_params.problem.vehicles.get(cl.vehicle_type, None)
        avg_speed = getattr(vt_spec, "avg_speed", 30.0) if vt_spec is not None else 30.0
        total_distance_km += cl.route_time * avg_speed

    dist_stats = distance_ratios(
        total_distance_km, num_customers, solution.total_vehicles
    )

    day_id = demand_path.stem
    result = {
        "day_id": day_id,
        "num_customers": num_customers,
        "total_demand_kg": total_demand_kg,
        "solver_cost": solver_cost,
        "fleet_type": fleet_type,
        "alpha": alpha,
        "C": C,
        "cost_per_drop": cost_per_d,
        "cost_per_kg": cost_per_k,
        "split_rate": split_r,
        "avg_visits_per_customer": avg_visits,
        # Derived, unit-free expressions
        "alpha_surcharge_pct": alpha_surcharge_pct,
        "c_pct_scv": c_pct_scv,
        # Detailed solution metrics
        "total_fixed_cost": solution.total_fixed_cost,
        "total_variable_cost": solution.total_variable_cost,
        "total_penalties": solution.total_penalties,
        "total_light_load_penalties": solution.total_light_load_penalties,
        "total_compartment_penalties": solution.total_compartment_penalties,
        "total_vehicles": solution.total_vehicles,
        "vehicles_used": solution.vehicles_used,
        "solver_runtime_sec": solution.solver_runtime_sec,
        "optimality_gap": solution.optimality_gap,
        "total_route_time_hours": total_route_time_hours,
        # New aggregated route metrics
        **rt_stats,
        **st_stats,
        "total_distance_km": total_distance_km,
        **dist_stats,
    }

    return _convert_numpy_types(result)  # type: ignore[no-any-return]
