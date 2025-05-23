"""Parse all MCVRP and CVRP JSON results in results/ and produce CSV summaries."""
import json
import csv
from pathlib import Path

from fleetmix.utils import PROJECT_ROOT

def parse_vrp_results(vrp_type: str):
    """Parse VRP results for a specific type (mcvrp or cvrp) and return rows."""
    results_dir = PROJECT_ROOT / "results"
    rows = []

    pattern = f"{vrp_type}_*.json"
    for json_file in sorted(results_dir.glob(pattern)):
        data = json.loads(json_file.read_text())
        summary = data.get("Solution Summary", {})
        
        # Extract instance name by removing the vrp_type prefix and any suffix
        instance = json_file.stem
        if vrp_type == "mcvrp":
            instance = instance.replace("mcvrp_", "")
        elif vrp_type == "cvrp":
            # Remove both "cvrp_" prefix and "_normal" suffix for CVRP instances
            instance = instance.replace("cvrp_", "").replace("_normal", "")
        
        used = int(summary.get("Total Vehicles", 0))
        expected = int(summary.get("Expected Vehicles", 0))
        rows.append({
            "Instance": instance,
            "Vehicles Used": used,
            "Expected Vehicles": expected,
            "Vehicles Difference": used - expected
        })

    return rows

def main():
    results_dir = PROJECT_ROOT / "results"
    
    # Parse MCVRP results
    mcvrp_rows = parse_vrp_results("mcvrp")
    if mcvrp_rows:
        mcvrp_csv = results_dir / "summary_mcvrp.csv"
        with mcvrp_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Instance",
                "Vehicles Used",
                "Expected Vehicles",
                "Vehicles Difference"
            ])
            writer.writeheader()
            writer.writerows(mcvrp_rows)
        print(f"Wrote MCVRP summary CSV to {mcvrp_csv}")
    else:
        print("No MCVRP results found")

    # Parse CVRP results
    cvrp_rows = parse_vrp_results("cvrp")
    if cvrp_rows:
        cvrp_csv = results_dir / "summary_cvrp.csv"
        with cvrp_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Instance",
                "Vehicles Used",
                "Expected Vehicles",
                "Vehicles Difference"
            ])
            writer.writeheader()
            writer.writerows(cvrp_rows)
        print(f"Wrote CVRP summary CSV to {cvrp_csv}")
    else:
        print("No CVRP results found")


if __name__ == "__main__":
    main() 