"""
Day-level optimization runner.
"""

from pathlib import Path

import pandas as pd

from fleetmix.api import optimize

# For derived parameter expressions
from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.experiments.alpha_analysis.metrics import (
    average_visits_per_customer,
    cost_per_drop,
    cost_per_kg,
    split_rate,
)
from fleetmix.utils.data_processing import load_customer_demand

BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config_experiments.yaml")


def run_day(
    demand_path: Path,
    fleet_params: FleetmixParams,
    fleet_type: str,
    alpha: float = 1.0,
    C: float = 0.0,
) -> dict:
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

    day_id = demand_path.stem
    return {
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
    }
