"""
save_results.py – centralised persistence + reporting utilities

This module is the **single exit point** for anything that needs to hit disk after an optimisation
run.  It intentionally concentrates all I/O – spreadsheets, JSON dumps, plots – so that the rest of
the codebase can remain side-effect free and therefore easier to test.

Key responsibilities
• Convert rich in-memory Python/Pandas objects into human-readable artefacts (Excel/JSON).
• Produce quick-look diagnostics such as demand/load histograms and interactive Folium maps.
• Guarantee the results directory structure exists and file names are timestamp-safe.

By funnelling output through one module we avoid scattered `to_excel` / `to_json` calls and keep the
pipeline composable – exactly the kind of "one obvious place" Jeff Dean and John Ousterhout
advocate for.
"""

import ast
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import seaborn as sns

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import (
    BenchmarkType,
    FleetmixSolution,
    VehicleConfiguration,
    VRPSolution,
)
from fleetmix.utils.logging import FleetmixLogger

logger = FleetmixLogger.get_logger(__name__)


def save_optimization_results(
    solution: FleetmixSolution,
    configurations: list[VehicleConfiguration],
    parameters: Parameters,
    filename: str | None = None,
    format: str = "excel",
    is_benchmark: bool = False,
    expected_vehicles: int | None = None,
) -> None:
    """Save optimization results to a file (Excel or JSON) and create visualization"""

    base_results_dir = parameters.results_dir

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".xlsx" if format == "excel" else ".json"
        output_filename = (
            base_results_dir / f"optimization_results_{timestamp}{extension}"
        )
    else:
        output_filename = Path(filename)

    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # Create a lookup dictionary for configurations
    config_lookup = {str(config.config_id): config for config in configurations}

    # Calculate metrics and prepare data
    if "Customers" in solution.selected_clusters.columns:
        # When split-stops are enabled the optimisation works with *pseudo* customers
        # (e.g. ``3::Dry`` and ``3::Frozen``) which means the raw list can contain
        # several entries for what is, conceptually, the *same* physical customer.
        # For reporting purposes we want to treat each origin customer only once so
        # that the "Customers per Cluster" statistics and the JSON/Excel exports
        # stay human-readable.
        if parameters.allow_split_stops:

            def _n_unique_origins(customers: list[str] | tuple[str, ...] | str) -> int:
                """Return the number of *origin* customers, ignoring goods suffixes."""
                if not isinstance(customers, (list, tuple)):
                    # If optimisation returned a single customer as a plain string
                    # we still count it as one origin.
                    return 1
                seen: set[str] = set()
                for cid in customers:
                    origin = cid.split("::")[0] if "::" in str(cid) else str(cid)
                    seen.add(origin)
                return len(seen)

            customers_per_cluster = solution.selected_clusters["Customers"].apply(
                _n_unique_origins
            )
        else:
            customers_per_cluster = solution.selected_clusters["Customers"].apply(len)
    else:
        # For benchmark results, use Num_Customers column
        customers_per_cluster = solution.selected_clusters["Num_Customers"]

    # Calculate load percentages
    load_percentages = []
    for _, cluster in solution.selected_clusters.iterrows():
        if "Vehicle_Utilization" in cluster:
            total_utilization = cluster["Vehicle_Utilization"]
        else:
            total_demand = (
                ast.literal_eval(cluster["Total_Demand"])
                if isinstance(cluster["Total_Demand"], str)
                else cluster["Total_Demand"]
            )
            config = config_lookup[str(cluster["Config_ID"])]
            total_utilization = (sum(total_demand.values()) / config.capacity) * 100

        load_percentages.append(total_utilization)

    load_percentages = pd.Series(load_percentages)

    summary_metrics = [
        ("Total Cost ($)", f"{solution.total_cost:,.2f}"),
        ("Fixed Cost ($)", f"{solution.total_fixed_cost:,.2f}"),
        ("Variable Cost ($)", f"{solution.total_variable_cost:,.2f}"),
        ("Total Penalties ($)", f"{solution.total_penalties:,.2f}"),
        ("  Light Load Penalties ($)", f"{solution.total_light_load_penalties:,.2f}"),
        (
            "  Compartment Setup Penalties ($)",
            f"{solution.total_compartment_penalties:,.2f}",
        ),
        ("Total Vehicles", solution.total_vehicles),
    ]

    if expected_vehicles is not None:
        summary_metrics.append(("Expected Vehicles", expected_vehicles))

    for vehicle_type in sorted(solution.vehicles_used.keys()):
        vehicle_count = solution.vehicles_used[vehicle_type]
        summary_metrics.append((f"Vehicles Type {vehicle_type}", vehicle_count))

    summary_metrics.extend(
        [
            ("Customers per Cluster (Min)", f"{customers_per_cluster.min():.0f}"),
            ("Customers per Cluster (Max)", f"{customers_per_cluster.max():.0f}"),
            ("Customers per Cluster (Avg)", f"{customers_per_cluster.mean():.1f}"),
            ("Customers per Cluster (Median)", f"{customers_per_cluster.median():.1f}"),
            ("Truck Load % (Min)", f"{load_percentages.min():.1f}"),
            ("Truck Load % (Max)", f"{load_percentages.max():.1f}"),
            ("Truck Load % (Avg)", f"{load_percentages.mean():.1f}"),
            ("Truck Load % (Median)", f"{load_percentages.median():.1f}"),
            ("---Parameters---", ""),
            ("Demand File", parameters.demand_file),
            ("Variable Cost per Hour", parameters.variable_cost_per_hour),
            ("Max Split Depth", parameters.clustering["max_depth"]),
            ("Clustering Method", parameters.clustering["method"]),
            ("Clustering Distance", parameters.clustering["distance"]),
            ("Geography Weight", parameters.clustering["geo_weight"]),
            ("Demand Weight", parameters.clustering["demand_weight"]),
            (
                "Route Time Estimation Method",
                parameters.clustering["route_time_estimation"],
            ),
            ("Light Load Penalty", parameters.light_load_penalty),
            ("Light Load Threshold", parameters.light_load_threshold),
            ("Compartment Setup Cost", parameters.compartment_setup_cost),
        ]
    )

    # Add vehicle types
    for v_type, specs in parameters.vehicles.items():
        summary_metrics.append((f"Vehicle Type {v_type} Capacity", specs.capacity))
        summary_metrics.append((f"Vehicle Type {v_type} Fixed Cost", specs.fixed_cost))
        summary_metrics.append((f"Vehicle Type {v_type} Avg Speed", specs.avg_speed))
        summary_metrics.append(
            (f"Vehicle Type {v_type} Service Time", specs.service_time)
        )
        summary_metrics.append(
            (f"Vehicle Type {v_type} Max Route Time", specs.max_route_time)
        )
        # If VehicleSpec has an 'extra' field, we want to include those:
        if hasattr(specs, "extra") and specs.extra:
            for extra_key, extra_value in specs.extra.items():
                summary_metrics.append(
                    (
                        f"Vehicle Type {v_type} {extra_key.replace('_', ' ').title()}",
                        extra_value,
                    )
                )

    # Prepare cluster details
    cluster_details = solution.selected_clusters.copy()
    if "Customers" in cluster_details.columns:
        # Deduplicate customer IDs for clearer reporting when split-stops are enabled
        if parameters.allow_split_stops:

            def _deduplicate_customer_ids(customers):
                """Return a list with at most one entry per *origin* customer."""
                if not isinstance(customers, (list, tuple)):
                    return customers
                unique: list[str] = []
                seen: set[str] = set()
                for cid in customers:
                    origin = cid.split("::")[0] if "::" in str(cid) else str(cid)
                    if origin not in seen:
                        unique.append(origin)
                        seen.add(origin)
                return unique

            cluster_details["Customers"] = cluster_details["Customers"].apply(
                _deduplicate_customer_ids
            )

        cluster_details["Num_Customers"] = cluster_details["Customers"].apply(len)
        cluster_details["Customers"] = cluster_details["Customers"].apply(str)
    if "TSP_Sequence" in cluster_details.columns:
        cluster_details["TSP_Sequence"] = cluster_details["TSP_Sequence"].apply(
            lambda x: " -> ".join(map(str, x))
            if isinstance(x, (list, tuple)) and x
            else str(x)
        )
    if "Total_Demand" in cluster_details.columns:
        cluster_details["Total_Demand"] = cluster_details["Total_Demand"].apply(str)

        # Add demand and load percentages by product type
        for cluster_idx, cluster in cluster_details.iterrows():
            config = config_lookup[str(cluster["Config_ID"])]

            total_demand = (
                ast.literal_eval(cluster["Total_Demand"])
                if isinstance(cluster["Total_Demand"], str)
                else cluster["Total_Demand"]
            )
            total_demand_sum = sum(total_demand.values())

            # Calculate demand percentage for each product type first
            for good in parameters.goods:
                demand_column_name = f"Demand_{good}_pct"
                cluster_details.at[cluster_idx, demand_column_name] = (
                    total_demand[good] / total_demand_sum if total_demand_sum > 0 else 0
                )

            # Then calculate load percentage for each product type
            for good in parameters.goods:
                load_column_name = f"Load_{good}_pct"
                cluster_details.at[cluster_idx, load_column_name] = (
                    total_demand[good] / config.capacity
                )

            # Calculate TOTAL load percentage and empty percentage
            config_capacity = config.capacity
            total_load_pct = (
                total_demand_sum / config_capacity if config_capacity > 0 else 0
            )
            cluster_details.at[cluster_idx, "Load_total_pct"] = total_load_pct
            cluster_details.at[cluster_idx, "Load_empty_pct"] = 1 - total_load_pct

    # Convert configurations to DataFrame for output compatibility
    configurations_df = pd.DataFrame([config.to_dict() for config in configurations])

    data = {
        "summary_metrics": summary_metrics,
        "configurations_df": configurations_df,
        "cluster_details": cluster_details,
        "vehicles_used": solution.vehicles_used,
        "other_considerations": [
            ("Total Vehicles Used", solution.total_vehicles),
            ("Number of Unserved Customers", len(solution.missing_customers)),
            (
                "Unserved Customers",
                str(list(solution.missing_customers))
                if solution.missing_customers
                else "None",
            ),
            (
                "Average Customers per Cluster",
                cluster_details["Num_Customers"].mean()
                if "Num_Customers" in cluster_details.columns
                and not cluster_details.empty
                else "N/A",
            ),
            (
                "Average Distance per Cluster",
                cluster_details["Estimated_Distance"].mean()
                if "Estimated_Distance" in cluster_details.columns
                and not cluster_details.empty
                else "N/A",
            ),
        ],
        "execution_details": [
            (
                "Execution Time (s)",
                next(
                    (
                        tm.wall_time
                        for tm in solution.time_measurements
                        if tm.span_name == "global"
                    ),
                    "N/A",
                )
                if solution.time_measurements
                else "N/A",
            ),
            ("Solver", solution.solver_name),
            ("Solver Status", solution.solver_status),
            ("Solver Runtime (s)", solution.solver_runtime_sec),
            ("Total Fixed Cost", solution.total_fixed_cost),
            ("Total Variable Cost", solution.total_variable_cost),
            ("Total Penalties", solution.total_penalties),
            ("Light Load Penalties", solution.total_light_load_penalties),
            ("Compartment Setup Penalties", solution.total_compartment_penalties),
            ("Total Cost", solution.total_cost),
            ("Demand File", parameters.demand_file),
        ],
    }

    # Process time measurements if provided
    time_measurements_for_excel = []
    time_measurements_for_json = {}
    if solution.time_measurements:
        for measurement in solution.time_measurements:
            time_measurements_for_excel.extend(
                [
                    (f"{measurement.span_name}_wall_time", measurement.wall_time),
                    (
                        f"{measurement.span_name}_process_user_time",
                        measurement.process_user_time,
                    ),
                    (
                        f"{measurement.span_name}_process_system_time",
                        measurement.process_system_time,
                    ),
                    (
                        f"{measurement.span_name}_children_user_time",
                        measurement.children_user_time,
                    ),
                    (
                        f"{measurement.span_name}_children_system_time",
                        measurement.children_system_time,
                    ),
                    (
                        f"{measurement.span_name}_total_cpu_time",
                        measurement.process_user_time
                        + measurement.process_system_time
                        + measurement.children_user_time
                        + measurement.children_system_time,
                    ),
                ]
            )
            # For JSON:
            measurement_values_for_json = asdict(measurement)
            span_name_for_json = measurement_values_for_json.pop("span_name")
            time_measurements_for_json[span_name_for_json] = measurement_values_for_json

    # Add to data dict for writing functions
    if time_measurements_for_excel:
        data["time_measurements_excel"] = (
            time_measurements_for_excel  # Used by _write_to_excel
        )
    if time_measurements_for_json:
        data["time_measurements_json"] = (
            time_measurements_for_json  # Used by _write_to_json
        )

    try:
        if format == "json":
            _write_to_json(
                str(output_filename), data
            )  # _write_to_json will use data['time_measurements_json']
        else:
            _write_to_excel(
                str(output_filename), data
            )  # _write_to_excel will use data['time_measurements_excel']

        # Only create visualization for optimization results
        if not is_benchmark:
            depot_coords = (parameters.depot["latitude"], parameters.depot["longitude"])
            visualize_clusters(
                solution.selected_clusters, depot_coords, str(output_filename)
            )

    except Exception as e:
        print(f"Error saving results to {output_filename}: {e!s}")
        raise


def _write_to_excel(filename: str, data: dict) -> None:
    """Write optimization results to Excel file."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Sheet 1: Summary
        pd.DataFrame(data["summary_metrics"], columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Solution Summary", index=False
        )

        # Sheet 2: Configurations
        data["configurations_df"].to_excel(
            writer, sheet_name="Configurations", index=False
        )

        # Sheet 3: Selected Clusters
        cluster_cols = [
            col
            for col in data["cluster_details"].columns
            if col not in ["Customers", "TSP_Sequence"]
        ] + ["Customers", "TSP_Sequence"]
        # Reorder cols to put Customers and TSP_Sequence last, if they exist
        cluster_cols_ordered = [
            col for col in cluster_cols if col in data["cluster_details"].columns
        ]
        data["cluster_details"].to_excel(
            writer,
            sheet_name="Selected Clusters",
            index=False,
            columns=cluster_cols_ordered,
        )

        # Sheet 4: Vehicle Usage
        vehicles_df = pd.DataFrame(
            [(k, v) for k, v in data["vehicles_used"].items()],
            columns=["Vehicle Type", "Count"],
        )
        vehicles_df.to_excel(writer, sheet_name="Vehicle Usage", index=False)

        # Sheet 5: Other Considerations
        pd.DataFrame(
            data["other_considerations"], columns=["Metric", "Value"]
        ).to_excel(writer, sheet_name="Other Considerations", index=False)

        # Sheet 6: Execution Details
        pd.DataFrame(data["execution_details"], columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Execution Details", index=False
        )

        # Sheet 7: Time Measurements (if available)
        if "time_measurements_excel" in data:  # MODIFIED key
            pd.DataFrame(
                data["time_measurements_excel"], columns=["Metric", "Value"]
            ).to_excel(writer, sheet_name="Time Measurements", index=False)


def _write_to_json(filename: str, data: dict) -> None:
    """Write optimization results to JSON file."""

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Convert vehicles_used to list of dictionaries
    # Ensure cluster details are serializable (TSP sequence might be list)
    serializable_clusters = data["cluster_details"].to_dict(orient="records")
    for cluster in serializable_clusters:
        if "TSP_Sequence" in cluster and isinstance(cluster["TSP_Sequence"], list):
            cluster["TSP_Sequence"] = " -> ".join(map(str, cluster["TSP_Sequence"]))

    vehicle_usage = [
        {"vehicle_type": k, "count": v} for k, v in data["vehicles_used"].items()
    ]

    json_data = {
        "Solution Summary": dict(data["summary_metrics"]),
        "Configurations": data["configurations_df"].to_dict(orient="records"),
        "Selected Clusters": serializable_clusters,  # Use serializable version
        "Vehicle Usage": vehicle_usage,
        "Other Considerations": dict(data["other_considerations"]),
        "Execution Details": dict(data["execution_details"]),
    }

    # Add time measurements if available
    if "time_measurements_json" in data:  # MODIFIED key
        json_data["Time Measurements"] = data["time_measurements_json"]

    with open(filename, "w") as f:
        json.dump(json_data, f, cls=NumpyEncoder, indent=2)


def visualize_clusters(
    selected_clusters: pd.DataFrame, depot_coords: tuple, filename: str
) -> None:
    """
    Create and save an interactive map visualization of the clusters in Bogotá.

    Args:
        selected_clusters: DataFrame containing cluster information
        depot_coords: Tuple of (latitude, longitude) coordinates for the depot
        filename: Base filename to save the plot (will append _clusters.html)
    """
    # Initialize the map centered on Bogotá
    m = folium.Map(
        location=[4.65, -74.1],  # Bogotá center
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # Create color palette for clusters
    n_clusters = len(selected_clusters)
    colors = sns.color_palette("husl", n_colors=n_clusters).as_hex()

    # Add depot marker
    folium.Marker(
        location=depot_coords,
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        popup="Depot",
    ).add_to(m)

    # Plot each cluster
    for idx, (_, cluster) in enumerate(selected_clusters.iterrows()):
        color = colors[idx]
        cluster_id = cluster["Cluster_ID"]
        config_id = cluster["Config_ID"]

        # Calculate total demand in kg
        total_demand = (
            sum(cluster["Total_Demand"].values())
            if isinstance(cluster["Total_Demand"], dict)
            else 0
        )
        if isinstance(cluster["Total_Demand"], str):
            total_demand = sum(ast.literal_eval(cluster["Total_Demand"]).values())

        # Get number of customers
        num_customers = len(
            ast.literal_eval(cluster["Customers"])
            if isinstance(cluster["Customers"], str)
            else cluster["Customers"]
        )

        # Prepare popup content with Method field
        popup_content = f"""
            <b>Cluster ID:</b> {cluster_id}<br>
            <b>Config ID:</b> {config_id}<br>
            <b>Method:</b> {cluster["Method"]}<br>
            <b>Customers:</b> {num_customers}<br>
            <b>Route Time:</b> {cluster["Route_Time"]:.2f} hrs<br>
            <b>Total Demand:</b> {total_demand:,.0f} kg
        """

        # Plot cluster centroid with larger circle
        folium.CircleMarker(
            location=(cluster["Centroid_Latitude"], cluster["Centroid_Longitude"]),
            radius=8,
            color=color,
            fill=True,
            popup=folium.Popup(popup_content, max_width=300),
            weight=2,
            fill_opacity=0.7,
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    viz_filename = str(filename).rsplit(".", 1)[0] + "_clusters.html"
    m.save(viz_filename)
