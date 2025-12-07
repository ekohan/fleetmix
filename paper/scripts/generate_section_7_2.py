"""
Script to generate Section 7.2 (Benefits of Using Multi-Compartment Vehicles) of the paper.
Replicates the analysis in "Benefits of Using Multi-Compartment Vehicles" of @paper/main.tex.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure plotting style
plt.style.use("classic")
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": "lightgrey",
        "grid.alpha": 0.5,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def extract_additional_metrics(data, row):
    """Extract additional metrics from JSON data"""

    selected_clusters = data.get("Selected Clusters", [])
    load_percentages = []
    customers_per_cluster = []
    route_times = []
    compartment_counts = []

    for cluster in selected_clusters:
        # Load percentage
        load_pct = cluster.get("Load_total_pct", 0) * 100
        if load_pct > 0:
            load_percentages.append(load_pct)

        # Customers per cluster
        num_customers = cluster.get("Num_Customers", 0)
        if num_customers > 0:
            customers_per_cluster.append(num_customers)

        # Route time
        route_time = cluster.get("Route_Time", 0)
        if route_time > 0:
            route_times.append(route_time)

        # Count compartments for MCV
        goods_in_config = cluster.get("Goods_In_Config", [])
        compartment_counts.append(len(goods_in_config))

    # Load statistics
    if load_percentages:
        # Match parse_benchmarking_results.py rounding behavior (early rounding)
        row["Avg Load %"] = float(f"{np.mean(load_percentages):.1f}")
    else:
        row["Avg Load %"] = 0.0

    # Customer distribution
    if customers_per_cluster:
        # Match parse_benchmarking_results.py rounding behavior (early rounding)
        row["Avg Customers per Vehicle"] = float(
            f"{np.mean(customers_per_cluster):.1f}"
        )
    else:
        row["Avg Customers per Vehicle"] = 0.0

    # Route time statistics
    if route_times:
        # Match parse_benchmarking_results.py rounding behavior (early rounding)
        row["Avg Route Time (hours)"] = float(f"{np.mean(route_times):.2f}")
    else:
        row["Avg Route Time (hours)"] = 0.0

    # Vehicle type counters
    row["Vehicles Type A"] = 0
    row["Vehicles Type B"] = 0
    row["Vehicles Type C"] = 0

    vehicle_usage = data.get("Vehicle Usage", [])
    for vehicle in vehicle_usage:
        vehicle_type = vehicle.get("vehicle_type", "")
        count = vehicle.get("count", 0)

        if vehicle_type.startswith("A") or vehicle_type == "A":
            row["Vehicles Type A"] = row.get("Vehicles Type A", 0) + count
        elif vehicle_type.startswith("B") or vehicle_type == "B":
            row["Vehicles Type B"] = row.get("Vehicles Type B", 0) + count
        elif vehicle_type.startswith("C") or vehicle_type == "C":
            row["Vehicles Type C"] = row.get("Vehicles Type C", 0) + count

    # MCV compartment analysis
    if compartment_counts:
        # Match parse_benchmarking_results.py rounding behavior (early rounding)
        row["Avg Compartments per Vehicle"] = float(
            f"{np.mean(compartment_counts):.2f}"
        )
    else:
        row["Avg Compartments per Vehicle"] = 0.0

    # Configuration parsing
    summary = data.get("Solution Summary", {})
    config_file = summary.get("Config File", "")

    # Determine vehicle type
    if "mcv" in config_file.lower():
        row["Vehicle Type"] = "MCV"
    elif "scv" in config_file.lower():
        row["Vehicle Type"] = "SCV"
    else:
        row["Vehicle Type"] = "Unknown"

    # Determine parameter type and variation
    if "baseline" in config_file.lower():
        row["Parameter Type"] = "Baseline"
        row["Variation Value"] = 0
    elif "capacity" in config_file.lower():
        row["Parameter Type"] = "Capacity"
        if "plus_20" in config_file.lower() or "p20" in config_file.lower():
            row["Variation Value"] = 20
        elif "plus_50" in config_file.lower() or "p50" in config_file.lower():
            row["Variation Value"] = 50
        elif "minus_20" in config_file.lower() or "m20" in config_file.lower():
            row["Variation Value"] = -20
        elif "minus_50" in config_file.lower() or "m50" in config_file.lower():
            row["Variation Value"] = -50
        else:
            row["Variation Value"] = 0
    elif "service_time" in config_file.lower():
        row["Parameter Type"] = "Service Time"
        if "plus_20" in config_file.lower() or "p20" in config_file.lower():
            row["Variation Value"] = 20
        elif "plus_50" in config_file.lower() or "p50" in config_file.lower():
            row["Variation Value"] = 50
        elif "minus_20" in config_file.lower() or "m20" in config_file.lower():
            row["Variation Value"] = -20
        elif "minus_50" in config_file.lower() or "m50" in config_file.lower():
            row["Variation Value"] = -50
        else:
            row["Variation Value"] = 0
    elif (
        "max_route_duration" in config_file.lower()
        or "max_route_time" in config_file.lower()
    ):
        row["Parameter Type"] = "Max Route Duration"
        if "plus_20" in config_file.lower() or "p20" in config_file.lower():
            row["Variation Value"] = 20
        elif "plus_50" in config_file.lower() or "p50" in config_file.lower():
            row["Variation Value"] = 50
        elif "minus_20" in config_file.lower() or "m20" in config_file.lower():
            row["Variation Value"] = -20
        elif "minus_50" in config_file.lower() or "m50" in config_file.lower():
            row["Variation Value"] = -50
        else:
            row["Variation Value"] = 0
    else:
        row["Parameter Type"] = "Unknown"
        row["Variation Value"] = 0

    return row


def parse_results(results_dir: Path):
    rows = []
    json_files = []

    # Walk through directory recursively
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(Path(root) / file)

    print(f"Found {len(json_files)} JSON files to parse.")

    for json_file in sorted(json_files):
        try:
            content = json_file.read_text()
            if not content.strip():
                continue

            data = json.loads(content)
            summary = data.get("Solution Summary", {})

            instance = json_file.stem
            used = int(summary.get("Total Vehicles", 0))

            total_cost = (
                summary.get("Total Cost ($)", "0").replace("$", "").replace(",", "")
            )
            try:
                total_cost = float(total_cost)
            except (ValueError, TypeError):
                total_cost = 0.0

            config_file = summary.get("Config File", "")

            row = {
                "Instance": instance,
                "Vehicles Used": used,
                "Total Cost ($)": total_cost,
                "Config File": config_file,
            }

            row = extract_additional_metrics(data, row)
            rows.append(row)

        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            continue

    return rows


def generate_plots(df, output_dir):
    print("Generating plots...")

    params = ["Capacity", "Max Route Duration", "Service Time"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plot_data_list = []

    # Include baseline data (variation = 0) for all parameters
    baseline_mcv = df[
        (df["Parameter Type"] == "Baseline") & (df["Vehicle Type"] == "MCV")
    ]
    baseline_scv = df[
        (df["Parameter Type"] == "Baseline") & (df["Vehicle Type"] == "SCV")
    ]

    for i, param in enumerate(params):
        param_data = df[df["Parameter Type"] == param].copy()

        # Group by variation and vehicle type
        summary = (
            param_data.groupby(["Variation Value", "Vehicle Type"])
            .agg({"Vehicles Used": "mean", "Total Cost ($)": "mean"})
            .reset_index()
        )

        # Add baseline point
        baseline_summary_mcv = pd.DataFrame(
            [
                {
                    "Variation Value": 0,
                    "Vehicle Type": "MCV",
                    "Vehicles Used": baseline_mcv["Vehicles Used"].mean(),
                    "Total Cost ($)": baseline_mcv["Total Cost ($)"].mean(),
                }
            ]
        )
        baseline_summary_scv = pd.DataFrame(
            [
                {
                    "Variation Value": 0,
                    "Vehicle Type": "SCV",
                    "Vehicles Used": baseline_scv["Vehicles Used"].mean(),
                    "Total Cost ($)": baseline_scv["Total Cost ($)"].mean(),
                }
            ]
        )

        summary = pd.concat(
            [summary, baseline_summary_mcv, baseline_summary_scv], ignore_index=True
        )
        summary = summary.sort_values("Variation Value")

        # Save data for table
        summary_copy = summary.copy()
        summary_copy["Parameter"] = param
        plot_data_list.append(summary_copy)

        # Fleet Size Plot (Top Row)
        ax_fleet = axes[0, i]
        mcv_data = summary[summary["Vehicle Type"] == "MCV"]
        scv_data = summary[summary["Vehicle Type"] == "SCV"]

        ax_fleet.plot(
            scv_data["Variation Value"],
            scv_data["Vehicles Used"],
            "s-",
            color="black",
            label="SCV",
            markerfacecolor="darkgrey",
            markersize=8,
        )
        ax_fleet.plot(
            mcv_data["Variation Value"],
            mcv_data["Vehicles Used"],
            "o:",
            color="black",
            label="MCV",
            markerfacecolor="white",
            markersize=8,
        )

        param_title = param
        if param == "Max Route Duration":
            param_title = "Max Route Duration"

        ax_fleet.set_title(f"{param_title} Variation (%)")
        ax_fleet.set_xlabel("Parameter Variation (%)")
        ax_fleet.set_ylabel("Number of Vehicles")
        ax_fleet.set_ylim(10, 50)
        ax_fleet.grid(True, alpha=0.3)
        ax_fleet.axvline(x=0, color="lightgrey", linestyle="-", alpha=0.5)
        ax_fleet.legend(loc="lower right")

        # Total Cost Plot (Bottom Row)
        ax_cost = axes[1, i]

        ax_cost.plot(
            scv_data["Variation Value"],
            scv_data["Total Cost ($)"],
            "s-",
            color="black",
            label="SCV",
            markerfacecolor="darkgrey",
            markersize=8,
        )
        ax_cost.plot(
            mcv_data["Variation Value"],
            mcv_data["Total Cost ($)"],
            "o:",
            color="black",
            label="MCV",
            markerfacecolor="white",
            markersize=8,
        )

        ax_cost.set_title(f"{param_title} Impact on Total Cost")
        ax_cost.set_xlabel("Parameter Variation (%)")
        ax_cost.set_ylabel("Total Cost ($)")
        ax_cost.set_ylim(3000, 6500)
        ax_cost.grid(True, alpha=0.3)
        ax_cost.axvline(x=0, color="lightgrey", linestyle="-", alpha=0.5)
        ax_cost.legend(loc="lower right")

    plt.tight_layout()
    output_file = output_dir / "figure_3_replicated.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Also save as PNG for README
    plt.savefig(output_dir / "figure_3_replicated.png", bbox_inches="tight", dpi=300)

    # Save table with Figure 3 values
    if plot_data_list:
        figure_3_df = pd.concat(plot_data_list, ignore_index=True)
        # Reorder columns
        cols = [
            "Parameter",
            "Variation Value",
            "Vehicle Type",
            "Vehicles Used",
            "Total Cost ($)",
        ]
        figure_3_df = figure_3_df[cols]

        output_csv = output_dir / "figure_3_values.csv"
        figure_3_df.to_csv(output_csv, index=False)
        print(f"Figure 3 values saved to {output_csv}")


def generate_table(df, output_dir):
    print("Generating table...")

    params = ["Capacity", "Service Time", "Max Route Duration"]
    variations = [-50, -20, 0, 20, 50]

    table_rows = []

    baseline_mcv = df[
        (df["Parameter Type"] == "Baseline") & (df["Vehicle Type"] == "MCV")
    ]
    baseline_scv = df[
        (df["Parameter Type"] == "Baseline") & (df["Vehicle Type"] == "SCV")
    ]

    for param in params:
        for var in variations:
            if var == 0:
                mcv_data = baseline_mcv
                scv_data = baseline_scv
            else:
                mcv_data = df[
                    (df["Parameter Type"] == param)
                    & (df["Variation Value"] == var)
                    & (df["Vehicle Type"] == "MCV")
                ]
                scv_data = df[
                    (df["Parameter Type"] == param)
                    & (df["Variation Value"] == var)
                    & (df["Vehicle Type"] == "SCV")
                ]

            row = {
                "Parameter": param,
                "Variation": f"{var}%",
                "MCV Load %": f"{mcv_data['Avg Load %'].mean():.1f}%",
                "SCV Load %": f"{scv_data['Avg Load %'].mean():.1f}%",
                "MCV Cust/Veh": f"{mcv_data['Avg Customers per Vehicle'].mean():.1f}",
                "SCV Cust/Veh": f"{scv_data['Avg Customers per Vehicle'].mean():.1f}",
                "MCV Route Duration": f"{mcv_data['Avg Route Time (hours)'].mean():.1f}",
                "SCV Route Duration": f"{scv_data['Avg Route Time (hours)'].mean():.1f}",
                "Type A": f"{mcv_data['Vehicles Type A'].mean():.1f}",
                "Type B": f"{mcv_data['Vehicles Type B'].mean():.1f}",
                "Type C": f"{mcv_data['Vehicles Type C'].mean():.1f}",
            }
            table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    output_csv = output_dir / "table_replicated.csv"
    table_df.to_csv(output_csv, index=False)
    print(f"Table saved to {output_csv}")

    return table_df


def create_readme(output_dir, table_df):
    readme_content = f"""# Benefits of Using Multi-Compartment Vehicles

Analysis replicated on {datetime.now().strftime("%Y-%m-%d")}.

## Figure 3: Average fleet size and total cost

![Figure 3](figure_3_replicated.png)

## Table: Average operational metrics

{table_df.to_markdown(index=False, disable_numparse=True)}

"""
    (output_dir / "README.md").write_text(readme_content)
    print(f"README.md created at {output_dir / 'README.md'}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_section_7_2.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    print(f"Processing results in {results_dir}")

    rows = parse_results(results_dir)
    if not rows:
        print("No data found.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Generate outputs
    generate_plots(df, results_dir)
    table_df = generate_table(df, results_dir)
    create_readme(results_dir, table_df)


if __name__ == "__main__":
    main()
