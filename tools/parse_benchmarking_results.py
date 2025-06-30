"""Parse all MCVRP, CVRP, and Case JSON results in results/ and produce CSV summaries."""

import csv
import json

from fleetmix.utils import PROJECT_ROOT


def parse_vrp_results(vrp_type: str):
    """Parse VRP results for a specific type (mcvrp, cvrp, or case) and return rows."""
    results_dir = PROJECT_ROOT / "results"
    rows = []

    # TODO: este pattern matching es horrible. simplificar.
    if vrp_type == "case":
        # For case benchmark, look for both default naming and config-based naming
        patterns = [
            "case_*.json",
            "*-*.json",
        ]  # case_instance.json or config-instance.json
        json_files = []
        for pattern in patterns:
            json_files.extend(results_dir.glob(pattern))
        # Remove duplicates and filter to only case-related files
        json_files = list(set(json_files))
        json_files = [
            f
            for f in json_files
            if "case_" in f.name
            or any(sales_prefix in f.name for sales_prefix in ["sales_", "demand_"])
        ]
    else:
        pattern = f"{vrp_type}_*.json"
        json_files = list(results_dir.glob(pattern))
        # Also look for config-based naming
        config_pattern = "*-*.json"
        config_files = [
            f
            for f in results_dir.glob(config_pattern)
            if (vrp_type == "mcvrp" and "mcvrp" in f.name)
            or (vrp_type == "cvrp" and "cvrp" in f.name and "mcvrp" not in f.name)
            or (vrp_type == "cvrp" and "X-n" in f.name and "mcvrp" not in f.name)
        ]
        json_files.extend(config_files)
        json_files = list(set(json_files))  # Remove duplicates

    for json_file in sorted(json_files):
        try:
            data = json.loads(json_file.read_text())
            summary = data.get("Solution Summary", {})

            # Extract instance name
            instance = json_file.stem

            # Handle different naming conventions
            if "-" in instance:
                # Config-based naming: config-instance or config-instance_suffix
                parts = instance.split("-", 1)
                if len(parts) == 2:
                    config_name = parts[0]
                    instance_name = parts[1]
                    if vrp_type == "cvrp" and instance_name.endswith("_normal"):
                        instance_name = instance_name.replace("_normal", "")
                    instance = instance_name
                else:
                    instance = instance
            else:
                # Traditional naming: vrp_type_instance
                if vrp_type == "mcvrp":
                    instance = instance.replace("mcvrp_", "")
                elif vrp_type == "cvrp":
                    instance = instance.replace("cvrp_", "").replace("_normal", "")
                elif vrp_type == "case":
                    instance = instance.replace("case_", "")

            # Extract data
            used = int(summary.get("Total Vehicles", 0))
            expected = (
                int(summary.get("Expected Vehicles", 0))
                if summary.get("Expected Vehicles", 0) != -1
                else 0
            )
            total_cost = (
                summary.get("Total Cost ($)", "").replace("$", "").replace(",", "")
                if summary.get("Total Cost ($)")
                else "0"
            )
            fixed_cost = (
                summary.get("Fixed Cost ($)", "").replace("$", "").replace(",", "")
                if summary.get("Fixed Cost ($)")
                else "0"
            )
            variable_cost = (
                summary.get("Variable Cost ($)", "").replace("$", "").replace(",", "")
                if summary.get("Variable Cost ($)")
                else "0"
            )
            config_file = summary.get("Config File", "default")

            # Clean up cost values
            try:
                total_cost = float(total_cost)
            except (ValueError, TypeError):
                total_cost = 0.0

            try:
                fixed_cost = float(fixed_cost)
            except (ValueError, TypeError):
                fixed_cost = 0.0

            try:
                variable_cost = float(variable_cost)
            except (ValueError, TypeError):
                variable_cost = 0.0

            row = {
                "Instance": instance,
                "Vehicles Used": used,
                "Expected Vehicles": expected,
                "Vehicles Difference": used - expected,
                "Total Cost ($)": f"{total_cost:.2f}",
                "Fixed Cost ($)": f"{fixed_cost:.2f}",
                "Variable Cost ($)": f"{variable_cost:.2f}",
                "Config File": config_file,
            }

            rows.append(row)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing {json_file}: {e}")
            continue

    return rows


def main():
    results_dir = PROJECT_ROOT / "results"

    fieldnames = [
        "Instance",
        "Vehicles Used",
        "Expected Vehicles",
        "Vehicles Difference",
        "Total Cost ($)",
        "Fixed Cost ($)",
        "Variable Cost ($)",
        "Config File",
    ]

    # Parse MCVRP results
    mcvrp_rows = parse_vrp_results("mcvrp")
    if mcvrp_rows:
        mcvrp_csv = results_dir / "summary_mcvrp.csv"
        with mcvrp_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mcvrp_rows)
        print(f"Wrote MCVRP summary CSV to {mcvrp_csv} ({len(mcvrp_rows)} instances)")
    else:
        print("No MCVRP results found")

    # Parse CVRP results
    cvrp_rows = parse_vrp_results("cvrp")
    if cvrp_rows:
        cvrp_csv = results_dir / "summary_cvrp.csv"
        with cvrp_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cvrp_rows)
        print(f"Wrote CVRP summary CSV to {cvrp_csv} ({len(cvrp_rows)} instances)")
    else:
        print("No CVRP results found")

    # Parse Case results
    case_rows = parse_vrp_results("case")
    if case_rows:
        case_csv = results_dir / "summary_case.csv"
        with case_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(case_rows)
        print(f"Wrote Case summary CSV to {case_csv} ({len(case_rows)} instances)")
    else:
        print("No Case results found")


if __name__ == "__main__":
    main()
