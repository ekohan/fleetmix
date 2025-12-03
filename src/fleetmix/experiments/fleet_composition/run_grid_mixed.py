"""
Grid executor for mixed-fleet (SCV+MCV) analysis.
Mirrors run_grid.py, reuses alpha_analysis config and metrics.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from fleetmix.api import optimize
from fleetmix.config import load_fleetmix_params
from fleetmix.experiments.fleet_composition.config import (
    ALPHA_GRID,
    C_VALUES,
    DEMAND_FILES,
)
from fleetmix.experiments.fleet_composition.fleet_templates import (
    make_mixed_fleet,
    make_scv_fleet,
)
from fleetmix.experiments.fleet_composition.metrics import (
    average_visits_per_customer,
    cost_per_drop,
    cost_per_kg,
    distance_ratios,
    route_time_stats,
    split_rate,
    stops_stats,
)
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.utils.logging import LogLevel, setup_logging

__all__ = ["convert_numpy_types", "main"]

PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PKG_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_RAW = RESULTS_DIR / "raw_mixed"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "summary_mixed.parquet"


def convert_numpy_types(obj):
    """Convert numpy types and complex objects to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif is_dataclass(obj):
        return convert_numpy_types(asdict(obj))  # type: ignore
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        conv = [convert_numpy_types(x) for x in obj]
        return conv if isinstance(obj, list) else tuple(conv)
    elif hasattr(obj, "__dict__"):
        return convert_numpy_types(obj.__dict__)
    return obj


def _vehicle_classification(params) -> Dict[str, str]:
    """
    Map vehicle type name -> 'MCV' or 'SCV' based on allowed_goods cardinality.
    """
    goods_set = set(params.problem.goods)
    mapping: Dict[str, str] = {}
    for name, spec in params.problem.vehicles.items():
        allowed = spec.allowed_goods
        if allowed is None:  # carries all goods -> MCV
            mapping[name] = "MCV"
        else:
            allowed_set = set(allowed)
            mapping[name] = (
                "MCV" if len(allowed_set) >= 2 or allowed_set == goods_set else "SCV"
            )
    return mapping


def _collect_day_summary(
    demand_path: Path, params, fleet_label: str, alpha: float, C: float
) -> Any:
    """Run optimization for one day and collect comprehensive metrics."""
    customers_df = load_customer_demand(str(demand_path))
    num_customers = len(customers_df)
    demand_cols = [c for c in customers_df.columns if c.endswith("_Demand")]
    total_kg = float(customers_df[demand_cols].sum().sum()) if demand_cols else 0.0

    solution = optimize(customers_df, params, output_dir=None)

    # Metrics
    cp_drop = cost_per_drop(solution.total_cost, num_customers)
    cp_kg = cost_per_kg(solution.total_cost, total_kg)
    sr = split_rate(solution)
    avg_visits = average_visits_per_customer(solution)

    # Vehicle composition
    vt_class = _vehicle_classification(params)  # vt_name -> 'MCV'/'SCV'
    used: Dict[str, int] = solution.vehicles_used or {}
    scv_used = sum(cnt for vt, cnt in used.items() if vt_class.get(vt, "SCV") == "SCV")
    mcv_used = sum(cnt for vt, cnt in used.items() if vt_class.get(vt, "SCV") == "MCV")
    total_used = scv_used + mcv_used
    mcv_share = float(mcv_used / total_used) if total_used > 0 else 0.0

    # Express α and C as percentages relative to base SCV fixed cost
    base_conf = load_fleetmix_params(
        Path("src/fleetmix/config/experiments/fleet_composition/base_config.yaml")
    )
    base_fc = float(next(iter(base_conf.problem.vehicles.values())).fixed_cost)
    alpha_pct = 100.0 * (alpha - 1.0)
    c_pct_scv = 100.0 * (C / base_fc) if base_fc else 0.0

    # Additional derived metrics
    vph = params.problem.variable_cost_per_hour
    total_route_time_hours = (
        float(solution.total_variable_cost / vph) if vph > 0 else 0.0
    )

    # Calculate vehicle utilization from selected clusters
    vehicle_utilizations = []
    scv_utilizations = []
    mcv_utilizations = []
    total_distance = (
        0.0  # We don't have exact distance but can estimate from route time
    )

    if solution.selected_clusters:
        # Create a mapping of config_id to capacity
        config_capacities = (
            {str(cfg.config_id): cfg.capacity for cfg in solution.configurations}
            if hasattr(solution, "configurations")
            else {}
        )

        # If configurations not in solution, get from params
        if not config_capacities:
            from fleetmix.utils.vehicle_configurations import (
                generate_vehicle_configurations,
            )

            configs = generate_vehicle_configurations(
                params.problem.vehicles, params.problem.goods
            )
            config_capacities = {str(cfg.config_id): cfg.capacity for cfg in configs}

        for cluster in solution.selected_clusters:
            # Calculate total demand weight for this cluster
            cluster_demand = (
                sum(cluster.total_demand.values()) if cluster.total_demand else 0.0
            )

            # Get vehicle capacity from config
            capacity = config_capacities.get(
                str(cluster.config_id), 1000
            )  # Default to 1000 if not found

            # Calculate utilization percentage
            utilization = (cluster_demand / capacity * 100) if capacity > 0 else 0.0
            vehicle_utilizations.append(utilization)

            # Track by vehicle type
            if vt_class.get(cluster.vehicle_type, "SCV") == "SCV":
                scv_utilizations.append(utilization)
            else:
                mcv_utilizations.append(utilization)

            # Estimate distance from route time (using avg speed)
            avg_speed = (
                params.problem.vehicles.get(cluster.vehicle_type, {}).avg_speed
                if hasattr(
                    params.problem.vehicles.get(cluster.vehicle_type, {}), "avg_speed"
                )
                else 30.0
            )
            total_distance += cluster.route_time * avg_speed

    # Calculate average utilizations
    avg_utilization = (
        float(np.mean(vehicle_utilizations)) if vehicle_utilizations else 0.0
    )
    avg_scv_utilization = float(np.mean(scv_utilizations)) if scv_utilizations else 0.0
    avg_mcv_utilization = float(np.mean(mcv_utilizations)) if mcv_utilizations else 0.0

    # Calculate utilization statistics
    utilization_stats = {
        "min": float(np.min(vehicle_utilizations)) if vehicle_utilizations else 0.0,
        "max": float(np.max(vehicle_utilizations)) if vehicle_utilizations else 0.0,
        "median": float(np.median(vehicle_utilizations))
        if vehicle_utilizations
        else 0.0,
        "std": float(np.std(vehicle_utilizations)) if vehicle_utilizations else 0.0,
    }

    # Route and stop stats using helper functions
    rt_stats = route_time_stats(solution)
    st_stats = stops_stats(solution)
    dist_stats = distance_ratios(total_distance, num_customers, total_used)

    # Extract runtime measurements
    time_measurements_data = []
    total_wall_time = 0.0
    if solution.time_measurements:
        for tm in solution.time_measurements:
            time_measurements_data.append(
                {
                    "span_name": tm.span_name,
                    "wall_time": float(tm.wall_time),
                    "process_user_time": float(tm.process_user_time),
                    "process_system_time": float(tm.process_system_time),
                    "children_user_time": float(tm.children_user_time),
                    "children_system_time": float(tm.children_system_time),
                }
            )
            total_wall_time += tm.wall_time

    return convert_numpy_types(
        {
            "instance": demand_path.stem,
            "fleet_type": fleet_label,  # "SCV_BASE" or "MIXED"
            "alpha": float(alpha),
            "C": float(C),
            "alpha_pct": float(alpha_pct),
            "C_pct_scv": float(c_pct_scv),
            "allow_split_stops": bool(params.problem.allow_split_stops),
            "total_cost": float(solution.total_cost),
            "total_vehicles": int(solution.total_vehicles),
            "vehicles_used": used,
            "scv_vehicles": int(scv_used),
            "mcv_vehicles": int(mcv_used),
            "mcv_share": float(mcv_share),
            "num_customers": int(num_customers),
            "total_demand": float(total_kg),
            "cost_per_drop": float(cp_drop),
            "cost_per_kg": float(cp_kg),
            "split_rate": float(sr),
            "average_visits_per_customer": float(avg_visits),
            "total_route_time_hours": float(total_route_time_hours),
            **rt_stats,
            **st_stats,
            # Vehicle utilization metrics
            "avg_vehicle_utilization_pct": float(avg_utilization),
            "avg_scv_utilization_pct": float(avg_scv_utilization),
            "avg_mcv_utilization_pct": float(avg_mcv_utilization),
            "utilization_min_pct": float(utilization_stats["min"]),
            "utilization_max_pct": float(utilization_stats["max"]),
            "utilization_median_pct": float(utilization_stats["median"]),
            "utilization_std_pct": float(utilization_stats["std"]),
            "total_distance_km": float(total_distance),
            **dist_stats,
            # Additional solution details
            "total_fixed_cost": float(solution.total_fixed_cost),
            "total_variable_cost": float(solution.total_variable_cost),
            "total_penalties": float(solution.total_penalties),
            "total_light_load_penalties": float(solution.total_light_load_penalties),
            "total_compartment_penalties": float(solution.total_compartment_penalties),
            "solver_runtime_sec": float(solution.solver_runtime_sec or 0.0),
            "solver_status": str(solution.solver_status or "Unknown"),
            "optimality_gap": float(solution.optimality_gap or 0.0),
            # Runtime measurements
            "total_runtime_sec": float(total_wall_time),
            "time_measurements": time_measurements_data,
        }
    )


def main() -> None:
    """Execute mixed fleet grid analysis."""
    # Ensure logs are informative for batch runs; respect env if set
    setup_logging()
    all_results = []

    print("Running mixed fleet analysis...")
    print(
        f"Grid size: {len(ALPHA_GRID)} alphas × {len(C_VALUES)} C values × {len(DEMAND_FILES)} demand days"
    )
    print(f"Total mixed runs: {len(ALPHA_GRID) * len(C_VALUES) * len(DEMAND_FILES)}")
    print(f"Total SCV baselines: {len(DEMAND_FILES)}")

    # 1) SCV baseline once per day
    print("\n=== Running SCV baselines ===")
    for demand_path in tqdm(DEMAND_FILES, desc="SCV baselines"):
        json_path = RESULTS_RAW / f"{demand_path.stem}_SCV_BASE.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            params = make_scv_fleet(demand_path.stem)
            # Promote debug logging if env requests it
            if params.runtime.debug:
                setup_logging(LogLevel.DEBUG)
            data = _collect_day_summary(
                demand_path, params, "SCV_BASE", alpha=1.0, C=0.0
            )
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        all_results.append(data)

    # 2) Mixed runs for every (α, C)
    print("\n=== Running mixed fleet grid ===")
    for demand_path in tqdm(DEMAND_FILES, desc="Demand days"):
        for alpha in tqdm(
            ALPHA_GRID, desc=f"Alpha values ({demand_path.stem})", leave=False
        ):
            for C in C_VALUES:
                json_path = (
                    RESULTS_RAW / f"{demand_path.stem}_MIXED_{alpha:.2f}_{C:.0f}.json"
                )
                if json_path.exists():
                    with open(json_path, "r") as f:
                        data = json.load(f)
                else:
                    params = make_mixed_fleet(
                        alpha=alpha, C=C, demand_day=demand_path.stem, allow_split=True
                    )
                    if params.runtime.debug:
                        setup_logging(LogLevel.DEBUG)
                    data = _collect_day_summary(
                        demand_path, params, "MIXED", alpha=alpha, C=C
                    )
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=2)
                all_results.append(data)

    print(f"\n=== Processing {len(all_results)} results ===")
    df = pd.DataFrame(all_results)

    # Optional: attach Δ% vs SCV baseline per (instance, α, C)
    baselines = df[df["fleet_type"] == "SCV_BASE"][["instance", "total_cost"]].rename(
        columns={"total_cost": "scv_cost"}
    )
    mixed = df[df["fleet_type"] == "MIXED"].merge(baselines, on="instance", how="left")
    mixed["delta_cost_pct_vs_scv"] = (
        100.0 * (mixed["total_cost"] - mixed["scv_cost"]) / mixed["scv_cost"]
    )

    # Combine mixed results with SCV baselines
    final_df = pd.concat([mixed, df[df["fleet_type"] == "SCV_BASE"]], ignore_index=True)

    # Save summary
    final_df.to_parquet(SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}")

    # Print basic statistics
    mixed_only = final_df[final_df["fleet_type"] == "MIXED"]
    if not mixed_only.empty:
        print("\n=== Summary Statistics ===")
        print(f"Mean MCV share: {mixed_only['mcv_share'].mean():.1%}")
        print(
            f"Mean cost improvement vs SCV: {mixed_only['delta_cost_pct_vs_scv'].mean():.1f}%"
        )
        print(
            f"Mixed fleet beats SCV in {(mixed_only['delta_cost_pct_vs_scv'] < 0).mean():.1%} of cases"
        )


if __name__ == "__main__":
    main()
