"""
Extract condensed raw results from full optimization JSONs.
"""

import json
from pathlib import Path
import argparse
import pandas as pd
from fleetmix.core_types import FleetmixSolution  # Import if needed for type hints

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _loads_if_str(value):
    """Return ``json.loads(value)`` if *value* is a ``str``
    otherwise pass the value through unchanged.

    The optimisation result JSON files sometimes serialise lists/dicts as
    true JSON strings (double-quoted) or embed them directly as proper
    Python/JSON objects.  Attempting to call ``json.loads`` on an already
    parsed object or on a string that does **not** contain valid JSON
    (e.g. uses single quotes) raises ``JSONDecodeError``.

    This helper makes the parsing tolerant:

    * If *value* is not a ``str`` we just return it.
    * If ``json.loads`` fails we fall back to returning the original string
      to avoid crashing the extraction pipeline.
    """

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Return the original string as a last resort.  Callers should handle
        # unexpected types defensively.
        return value


def extract_metrics(full_json: dict, fleet_type: str, alpha: float = 1.0, C: float = 0.0) -> dict:
    """Extract key metrics from full optimization JSON."""
    summary = full_json.get('Solution Summary', {})
    execution = full_json.get('Execution Details', {})
    
    # Compute basics
    clusters = full_json.get('Selected Clusters', [])

    # ``Customers`` and ``Total_Demand`` fields might already be proper objects
    # or JSON-encoded strings.  We use the helper above to read them safely.
    num_customers = sum(len(_loads_if_str(cluster.get('Customers', []))) for cluster in clusters)

    total_demand_kg = 0.0
    for cluster in clusters:
        total_dem_json = _loads_if_str(cluster.get('Total_Demand', {}))
        if isinstance(total_dem_json, dict):
            total_demand_kg += sum(total_dem_json.values())
    solver_cost = float(summary.get('Total Cost ($)', '0.0').replace(',', ''))
    
    # Cost breakdowns
    total_fixed_cost = float(summary.get('Fixed Cost ($)', '0.0').replace(',', ''))
    total_variable_cost = float(summary.get('Variable Cost ($)', '0.0').replace(',', ''))
    total_penalties = float(summary.get('Total Penalties ($)', '0.0').replace(',', ''))
    total_light_load_penalties = float(summary.get('  Light Load Penalties ($)', '0.0').replace(',', ''))
    total_compartment_penalties = float(summary.get('  Compartment Setup Penalties ($)', '0.0').replace(',', ''))
    
    # Vehicle metrics
    total_vehicles = int(summary.get('Total Vehicles', 0))
    vehicles_used = {}
    for key in summary:
        if key.startswith('Vehicles Type '):
            vehicles_used[key.replace('Vehicles Type ', '')] = int(summary[key])
    
    # Runtime and gap
    solver_runtime_sec = float(execution.get('Solver Runtime (s)', 0.0))
    optimality_gap = float(execution.get('Optimality Gap (%)', 0.0))
    
    # Route time
    variable_cost_per_hour = float(summary.get('Variable Cost per Hour', 10.0))
    total_route_time_hours = total_variable_cost / variable_cost_per_hour if variable_cost_per_hour > 0 else 0.0
    
    # Derived metrics (placeholders; compute if needed)
    cost_per_drop = solver_cost / num_customers if num_customers > 0 else 0.0
    cost_per_kg = solver_cost / total_demand_kg if total_demand_kg > 0 else 0.0
    split_rate = 0.0  # Placeholder; compute from clusters if needed
    avg_visits_per_customer = 1.0  # Placeholder
    
    # Base SCV fixed cost (hardcoded from your examples; adjust if needed)
    base_fixed_cost = 100.0
    alpha_surcharge_pct = 100 * (alpha - 1)
    c_pct_scv = 100 * (C / base_fixed_cost) if base_fixed_cost else 0.0
    
    # Day ID from demand file
    day_id = Path(summary.get('Demand File', 'unknown')).stem
    
    return {
        "day_id": day_id,
        "num_customers": num_customers,
        "total_demand_kg": total_demand_kg,
        "solver_cost": solver_cost,
        "fleet_type": fleet_type,
        "alpha": alpha,
        "C": C,
        "cost_per_drop": cost_per_drop,
        "cost_per_kg": cost_per_kg,
        "split_rate": split_rate,
        "avg_visits_per_customer": avg_visits_per_customer,
        "alpha_surcharge_pct": alpha_surcharge_pct,
        "c_pct_scv": c_pct_scv,
        "total_fixed_cost": total_fixed_cost,
        "total_variable_cost": total_variable_cost,
        "total_penalties": total_penalties,
        "total_light_load_penalties": total_light_load_penalties,
        "total_compartment_penalties": total_compartment_penalties,
        "total_vehicles": total_vehicles,
        "vehicles_used": vehicles_used,
        "solver_runtime_sec": solver_runtime_sec,
        "optimality_gap": optimality_gap,
        "total_route_time_hours": total_route_time_hours,
    }

def _infer_fleet_type(full_json: dict) -> str:
    """Heuristically determine whether result corresponds to an MCV or SCV run.

    We treat it as SCV if **any** vehicle type keys (either in the Solution
    Summary or Configurations list) start with the prefix ``SCV`` – otherwise
    we fall back to MCV.
    """

    summary_keys = full_json.get("Solution Summary", {}).keys()
    for k in summary_keys:
        if k.startswith("Vehicles Type SCV"):
            return "SCV"

    for cfg in full_json.get("Configurations", []):
        vt = cfg.get("Vehicle_Type", "")
        if str(vt).startswith("SCV"):
            return "SCV"

    return "MCV"


def process_directory(input_dir: Path, output_dir: Path, fleet_type: str, alpha: float, C: float):
    """Process all JSON files in input_dir and save extracted versions to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for json_path in input_dir.glob('*.json'):
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        this_fleet_type = fleet_type if fleet_type != "auto" else _infer_fleet_type(full_data)

        extracted = extract_metrics(full_data, this_fleet_type, alpha, C)
        
        # Generate output filename similar to raw_results
        # Use the original input file's stem to guarantee a 1-to-1 mapping
        # between input and output files. Relying solely on ``day_id`` can
        # cause collisions when multiple optimisation runs use the same
        # demand file. Example: several runs with different parameters but
        # the same day all map to ``sales_2024_avg_day_demand``.

        input_stem = json_path.stem
        output_filename = f"{input_stem}_{this_fleet_type}_{alpha:.2f}_{C:.0f}.json"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(extracted, f, indent=2)
        
        print(f"Processed {json_path} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract raw results from full optimization JSONs.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing full JSONs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for extracted JSONs')
    parser.add_argument('--fleet_type', type=str, default='auto', help='Fleet type: MCV, SCV, or auto to infer per file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value')
    parser.add_argument('--C', type=float, default=0.0, help='C value')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    process_directory(input_dir, output_dir, args.fleet_type.lower(), args.alpha, args.C)

if __name__ == "__main__":
    main()