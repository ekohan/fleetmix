"""Parse all mcvrp JSON results in results/ and produce a CSV summary."""
import json
import csv
from pathlib import Path

from fleetmix.utils import PROJECT_ROOT

def main():
    results_dir = PROJECT_ROOT / "results"
    out_csv = results_dir / "summary_mcvrp.csv"
    rows = []

    for json_file in sorted(results_dir.glob("*mcvrp_*.json")):
        data = json.loads(json_file.read_text())
        summary = data.get("Solution Summary", {})
        instance = json_file.stem.replace("mcvrp_", "")
        used = int(summary.get("Total Vehicles", 0))
        expected = int(summary.get("Expected Vehicles", 0))
        rows.append({
            "Instance": instance,
            "Vehicles Used": used,
            "Expected Vehicles": expected,
            "Vehicles Difference": used - expected
        })

    # Write out CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Instance",
            "Vehicles Used",
            "Expected Vehicles",
            "Vehicles Difference"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary CSV to {out_csv}")


if __name__ == "__main__":
    main() 