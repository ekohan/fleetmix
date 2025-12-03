import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def parse_filename(filename):
    """
    Parses filename to extract year (benchmark) and category.
    Expected format: mcvrp_<YEAR>_<CATEGORY>_(<ID>).json
    Example: mcvrp_2015_10_3_3_1_(01).json
    """
    # Remove extension
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    if len(parts) < 4:
        return None, None

    # Check prefix
    if parts[0] != "mcvrp":
        return None, None

    year = parts[1]

    # The last part is the ID, e.g. "(01)"
    # The category is everything between year and ID
    category_parts = parts[2:-1]
    category = "_".join(category_parts)

    return year, category


def get_benchmark_label(year):
    if year == "2015":
        return "Henke et al. (2015)"
    elif year == "2019":
        return "Henke et al. (2019)"
    return year


def process_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        summary = data.get("Solution Summary", {})
        exec_details = data.get("Execution Details", {})
        time_measurements = data.get("Time Measurements", {})

        total_vehicles = summary.get("Total Vehicles")
        expected_vehicles = summary.get("Expected Vehicles")

        # Try to get runtime
        runtime = exec_details.get("Execution Time (s)")
        if runtime is None:
            runtime = time_measurements.get("global", {}).get("wall_time")

        if (
            total_vehicles is not None
            and expected_vehicles is not None
            and runtime is not None
        ):
            return {
                "total_vehicles": int(total_vehicles),
                "expected_vehicles": int(expected_vehicles),
                "runtime": float(runtime),
            }
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate MCVRP comparison table.")
    parser.add_argument("directory", help="Directory containing JSON result files")
    args = parser.parse_args()

    target_dir = args.directory
    # If path is relative, assume it's from project root (cwd)
    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)

    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.", file=sys.stderr)
        sys.exit(1)

    files = glob.glob(os.path.join(target_dir, "*.json"))

    # (benchmark_label, category) -> list of results
    aggregated_data = defaultdict(list)

    for filepath in files:
        filename = os.path.basename(filepath)

        if filename.endswith("_dummy.json"):
            continue

        year, category = parse_filename(filename)
        if not year or not category:
            continue

        result = process_file(filepath)
        if result:
            benchmark = get_benchmark_label(year)
            aggregated_data[(benchmark, category)].append(result)

    # Sort keys: 2015 first, then 2019
    sorted_keys = sorted(aggregated_data.keys(), key=lambda k: (k[0], k[1]))

    output_path = os.path.join(target_dir, "README.md")

    with open(output_path, "w") as f:
        # Print Markdown Table to file
        f.write(
            "| Benchmark | Category | Instances | # Same vehicles | # Fewer vehicles | Run time (s) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")

        last_benchmark = None

        for benchmark, category in sorted_keys:
            results = aggregated_data[(benchmark, category)]

            count = len(results)
            same = sum(
                1 for r in results if r["total_vehicles"] == r["expected_vehicles"]
            )
            fewer = sum(
                1 for r in results if r["total_vehicles"] < r["expected_vehicles"]
            )
            avg_runtime = (
                statistics.mean(r["runtime"] for r in results) if results else 0
            )

            display_benchmark = benchmark if benchmark != last_benchmark else ""
            last_benchmark = benchmark

            f.write(
                f"| {display_benchmark} | {category} | {count} | {same} | {fewer} | {avg_runtime:.1f} |\n"
            )

    print(f"Table generated at: {output_path}")


if __name__ == "__main__":
    main()
