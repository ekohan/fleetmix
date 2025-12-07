"""
Streamlit GUI for fleetmix optimizer.
All GUI code in one file to minimize changes.
"""

import dataclasses
import json
import multiprocessing
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import streamlit as st

from fleetmix import api
from fleetmix.config import FleetmixParams, load_fleetmix_params
from fleetmix.core_types import DepotLocation, FleetmixSolution, VehicleSpec

# Page configuration
st.set_page_config(
    page_title="Fleetmix Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional styling */
    .stButton button {
        background-color: #0068c9;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #0051a2;
        transform: translateY(-1px);
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .error-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "optimization_running" not in st.session_state:
        st.session_state.optimization_running = False
    if "parameters" not in st.session_state:
        # Try to load default config
        default_config_path = Path("src/fleetmix/config/default_config.yaml")
        if default_config_path.exists():
            st.session_state.parameters = load_fleetmix_params(default_config_path)
        else:
            # Fallback logic or error if critical
            st.error("Default configuration not found.")
            st.session_state.parameters = None
    if "error_info" not in st.session_state:
        st.session_state.error_info = None


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif is_dataclass(obj) and not isinstance(obj, type):
        return convert_numpy_types(asdict(obj))
    return obj


def run_optimization_in_process(
    demand_path: str,
    params_source: Union[FleetmixParams, str, Path],
    output_dir: str,
    status_file: str,
) -> FleetmixSolution | None:
    """Runs optimization in separate process to support multiprocessing."""
    try:
        # Update status
        with open(status_file, "w") as f:
            json.dump({"stage": "Initializing...", "progress": 0}, f)

        # Obtain FleetmixParams
        if isinstance(params_source, (str, Path)):
            params = load_fleetmix_params(str(params_source))
        else:
            params = params_source

        # Update status - generating clusters
        with open(status_file, "w") as f:
            json.dump({"stage": "Generating clusters...", "progress": 25}, f)

        # Set output directory in params
        params = dataclasses.replace(
            params, io=dataclasses.replace(params.io, results_dir=Path(output_dir))
        )

        # Run optimization
        solution = api.optimize(
            demand=demand_path,
            config=params,
            output_dir=output_dir,
            format="json",
            verbose=False,
        )

        # Update status - optimization complete
        # Use atomic write (write to temp then rename) to prevent read race conditions
        status_path = Path(status_file)
        temp_status_path = status_path.with_suffix(".tmp")
        with open(temp_status_path, "w") as f:
            json.dump(
                {
                    "stage": "Optimization complete!",
                    "progress": 100,
                    "solution": convert_numpy_types(solution),
                },
                f,
            )
        temp_status_path.replace(status_path)

        return solution

    except Exception as e:
        # Write error to status file
        status_path = Path(status_file)
        temp_status_path = status_path.with_suffix(".tmp")
        with open(temp_status_path, "w") as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
        temp_status_path.replace(status_path)
        raise


def collect_parameters_from_ui() -> FleetmixParams:
    """Build FleetmixParams object from Streamlit widgets."""

    # Start with current parameters in session state
    params: FleetmixParams = st.session_state.parameters

    # Collect UI overrides from session_state keys
    # (We manually handle updates below, assuming st.session_state.parameters is the source of truth
    # and widgets update directly or we read widget values)

    # Ideally, we read values from the widgets which have keys.
    # However, Streamlit widgets with keys automatically update session_state[key].
    # We can iterate over relevant keys and update 'params'.

    # Problem Updates
    new_problem = params.problem

    # Vehicles
    if "param_vehicles" in st.session_state:
        # This is a bit tricky as param_vehicles is a dict of VehicleSpecs or dicts
        # We need to ensure we have VehicleSpecs
        vehicles_data = st.session_state["param_vehicles"]
        new_vehicles = {}
        for name, data in vehicles_data.items():
            if isinstance(data, VehicleSpec):
                new_vehicles[name] = data
            elif isinstance(data, dict):
                # Reconstruct VehicleSpec
                # Important: handle allowed_goods correctly
                new_vehicles[name] = VehicleSpec(
                    capacity=data["capacity"],
                    fixed_cost=data["fixed_cost"],
                    avg_speed=data["avg_speed"],
                    service_time=data["service_time"],
                    max_route_time=data["max_route_time"],
                    allowed_goods=data.get("allowed_goods"),
                    compartments=data.get("compartments", {}),
                    extra=data.get("extra", {}),
                )
        new_problem = dataclasses.replace(new_problem, vehicles=new_vehicles)

    # Depot location
    if (
        "param_depot_latitude" in st.session_state
        and "param_depot_longitude" in st.session_state
    ):
        new_depot = DepotLocation(
            latitude=st.session_state["param_depot_latitude"],
            longitude=st.session_state["param_depot_longitude"],
        )
        new_problem = dataclasses.replace(new_problem, depot=new_depot)

    # Other problem params
    problem_fields = {
        "param_variable_cost_per_hour": "variable_cost_per_hour",
        "param_light_load_penalty": "light_load_penalty",
        "param_light_load_threshold": "light_load_threshold",
        "param_compartment_setup_cost": "compartment_setup_cost",
        "param_allow_split_stops": "allow_split_stops",
    }

    problem_updates = {}
    for key, field in problem_fields.items():
        if key in st.session_state:
            problem_updates[field] = st.session_state[key]

    if problem_updates:
        new_problem = dataclasses.replace(new_problem, **problem_updates)

    # Always update problem in params
    params = dataclasses.replace(params, problem=new_problem)

    # Algorithm Updates
    new_algorithm = params.algorithm
    algorithm_fields = {
        "param_clustering_method": "clustering_method",
        "param_route_time_estimation": "route_time_estimation",
        "param_clustering_max_depth": "clustering_max_depth",
        "param_post_optimization": "post_optimization",
        "param_small_cluster_size": "small_cluster_size",
        "param_nearest_merge_candidates": "nearest_merge_candidates",
        "param_max_improvement_iterations": "max_improvement_iterations",
    }

    algorithm_updates = {}
    for key, field in algorithm_fields.items():
        if key in st.session_state:
            algorithm_updates[field] = st.session_state[key]

    if algorithm_updates:
        new_algorithm = dataclasses.replace(new_algorithm, **algorithm_updates)
        params = dataclasses.replace(params, algorithm=new_algorithm)

    # IO params are mostly handled by file uploads / defaults,
    # but we update demand file path in the main loop before running.

    st.session_state.parameters = params
    return params


def display_results(solution: dict[str, Any], output_dir: Path) -> None:
    """Display optimization results."""
    st.success("✅ Optimization completed successfully!")

    # Calculate total cost
    total_cost = (
        solution.get("total_fixed_cost", 0)
        + solution.get("total_variable_cost", 0)
        + solution.get("total_penalties", 0)
    )

    # Get total execution time from time_measurements if available
    # time_measurements is a list of TimeMeasurement dicts with span_name field
    total_time = solution.get("solver_runtime_sec", 0)
    time_measurements = solution.get("time_measurements")
    if time_measurements and isinstance(time_measurements, list):
        for tm in time_measurements:
            if isinstance(tm, dict) and tm.get("span_name") == "global":
                total_time = tm.get("wall_time", total_time)
                break

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col2:
        st.metric(
            "Vehicles Used",
            solution.get("total_vehicles", len(solution.get("vehicles_used", {}))),
        )
    with col3:
        st.metric("Missing Customers", len(solution.get("missing_customers", [])))
    with col4:
        st.metric("Total Time", f"{total_time:.1f}s")

    # Cost breakdown
    st.subheader("📊 Cost Breakdown")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fixed Cost", f"${solution.get('total_fixed_cost', 0):,.2f}")
    with col2:
        st.metric("Variable Cost", f"${solution.get('total_variable_cost', 0):,.2f}")
    with col3:
        st.metric("Penalties", f"${solution.get('total_penalties', 0):,.2f}")

    # Fleet efficiency metrics
    selected_clusters = solution.get("selected_clusters", [])
    total_vehicles = solution.get("total_vehicles", 0)
    configurations = solution.get("configurations", [])

    # Build a lookup for vehicle capacities by config_id
    config_capacity = {}
    for cfg in configurations:
        if isinstance(cfg, dict):
            cfg_id = str(cfg.get("config_id", ""))
            capacity = cfg.get("capacity", 0)
            if cfg_id and capacity:
                config_capacity[cfg_id] = capacity

    if selected_clusters and total_vehicles > 0:
        # Calculate metrics from cluster data
        load_percentages = []
        total_customers = 0

        for cluster in selected_clusters:
            if isinstance(cluster, dict):
                # Count customers from the customers list
                customers_list = cluster.get("customers", [])
                num_customers = (
                    len(customers_list) if isinstance(customers_list, list) else 0
                )
                total_customers += num_customers

                # Calculate load percentage from total_demand and vehicle capacity
                total_demand = cluster.get("total_demand", {})
                config_id = str(cluster.get("config_id", ""))
                capacity = config_capacity.get(config_id, 0)

                if isinstance(total_demand, dict) and capacity > 0:
                    demand_sum = sum(total_demand.values())
                    load_pct = (demand_sum / capacity) * 100
                    load_percentages.append(load_pct)

        st.subheader("📈 Fleet Efficiency")
        col1, col2, col3 = st.columns(3)
        with col1:
            if load_percentages:
                avg_load = sum(load_percentages) / len(load_percentages)
                st.metric("Avg Truck Load", f"{avg_load:.1f}%")
            else:
                st.metric("Avg Truck Load", "N/A")
        with col2:
            customers_per_vehicle = total_customers / total_vehicles
            st.metric("Customers per Vehicle", f"{customers_per_vehicle:.1f}")
        with col3:
            optimality_gap = solution.get("optimality_gap")
            if optimality_gap is not None and optimality_gap > 0:
                st.metric("Optimality Gap", f"{optimality_gap:.2f}%")
            else:
                st.metric("Solution Quality", "Optimal")

    # Vehicle usage details
    if solution.get("vehicles_used"):
        st.subheader("🚚 Vehicle Usage")
        vehicle_list_for_df = []
        usage_data = solution["vehicles_used"]

        if isinstance(usage_data, dict):
            vehicle_list_for_df = [
                {"Vehicle Type": k, "Count": v} for k, v in usage_data.items()
            ]
        elif isinstance(usage_data, list):
            for item in usage_data:
                if isinstance(item, dict):
                    vtype = item.get("vehicle_type") or item.get("type")
                    count = item.get("count")
                    if vtype is not None:
                        vehicle_list_for_df.append(
                            {"Vehicle Type": vtype, "Count": count}
                        )

        if vehicle_list_for_df:
            st.dataframe(pd.DataFrame(vehicle_list_for_df), use_container_width=True)

    # Download section
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)

    excel_files = list(output_dir.glob("optimization_results_*.xlsx"))
    json_files = list(output_dir.glob("optimization_results_*.json"))

    with col1:
        if excel_files:
            with open(excel_files[0], "rb") as f:
                st.download_button(
                    label="Download Excel Results",
                    data=f.read(),
                    file_name=excel_files[0].name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    with col2:
        if json_files:
            with open(json_files[0], "rb") as f:
                st.download_button(
                    label="Download JSON Results",
                    data=f.read(),
                    file_name=json_files[0].name,
                    mime="application/json",
                )

    # Display map if available
    html_files = list(output_dir.glob("optimization_results_*_clusters.html"))
    if html_files:
        st.subheader("🗺️ Cluster Visualization")
        try:
            with open(html_files[0]) as f:
                st.components.v1.html(f.read(), height=600)
        except Exception as e:
            st.error(f"Could not load map: {e}")


def main() -> None:
    """Main Streamlit app."""
    init_session_state()

    st.title("🚚 Fleetmix Optimizer")
    st.markdown("Optimize your fleet size and mix for heterogeneous vehicle routing")

    # --- Sidebar ---
    with st.sidebar:
        st.header("📋 Configuration")

        # 1. Upload Demand Data
        st.subheader("1. Upload Demand Data")
        uploaded_demand = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="Upload a CSV file with customer demand data",
            key="demand_uploader",
        )
        if uploaded_demand is not None:
            st.session_state.uploaded_data = pd.read_csv(uploaded_demand)
            st.success(f"✓ Loaded {len(st.session_state.uploaded_data)} customers")

        # 2. Upload Config (Optional)
        st.subheader("2. Upload Config (Optional)")
        uploaded_config = st.file_uploader(
            "Choose YAML file",
            type=["yaml", "yml"],
            help="Upload a YAML configuration file",
            key="config_uploader",
        )

        if uploaded_config is not None:
            # Load config from uploaded file
            try:
                # Save to temp file to use load_fleetmix_params
                with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
                    tmp.write(uploaded_config.getvalue())
                    tmp_path = Path(tmp.name)

                loaded_params = load_fleetmix_params(tmp_path)
                st.session_state.parameters = loaded_params
                st.success("✓ Configuration loaded!")
                Path(tmp.name).unlink()
            except Exception as e:
                st.error(f"Error loading config: {e}")

        # 3. Parameters Editing
        st.subheader("3. Configure Parameters")

        params = st.session_state.parameters
        if params is None:
            st.error("No configuration loaded.")
            return

        with st.expander("🚚 Vehicles", expanded=False):
            st.markdown("**Configure Vehicle Types**")

            # Access vehicles from problem params
            current_vehicles = params.problem.vehicles

            # We need to keep track of edited vehicles.
            # We'll create a copy for editing.
            edited_vehicles = {}

            available_goods = params.problem.goods

            for v_name, v_spec in current_vehicles.items():
                st.markdown(f"**{v_name}**")
                col1, col2 = st.columns(2)
                cap = col1.number_input(
                    f"Capacity ({v_name})",
                    value=int(v_spec.capacity),
                    step=100,
                    key=f"v_cap_{v_name}",
                )
                cost = col2.number_input(
                    f"Fixed Cost ({v_name})",
                    value=int(v_spec.fixed_cost),
                    step=10,
                    key=f"v_cost_{v_name}",
                )

                col3, col4, col5 = st.columns(3)
                speed = col3.number_input(
                    f"Speed ({v_name})",
                    value=float(v_spec.avg_speed),
                    step=5.0,
                    key=f"v_speed_{v_name}",
                )
                service = col4.number_input(
                    f"Service Time ({v_name})",
                    value=float(v_spec.service_time),
                    step=5.0,
                    key=f"v_service_{v_name}",
                )
                route_time = col5.number_input(
                    f"Max Time ({v_name})",
                    value=float(v_spec.max_route_time),
                    step=1.0,
                    key=f"v_time_{v_name}",
                )

                default_goods = (
                    v_spec.allowed_goods if v_spec.allowed_goods else available_goods
                )
                goods = st.multiselect(
                    f"Allowed Goods ({v_name})",
                    options=available_goods,
                    default=default_goods,
                    key=f"v_goods_{v_name}",
                )
                if not goods:
                    goods = available_goods  # Default to all if empty

                # Reconstruct spec
                edited_vehicles[v_name] = dataclasses.replace(
                    v_spec,
                    capacity=cap,
                    fixed_cost=cost,
                    avg_speed=speed,
                    service_time=service,
                    max_route_time=route_time,
                    allowed_goods=goods,
                )

            # Save to session state to be picked up by collect_parameters
            st.session_state.param_vehicles = edited_vehicles

        with st.expander("⚙️ Operations", expanded=False):
            st.markdown("**Depot Location**")
            col1, col2 = st.columns(2)
            col1.number_input(
                "Latitude",
                value=float(params.problem.depot.latitude),
                format="%.4f",
                key="param_depot_latitude",
            )
            col2.number_input(
                "Longitude",
                value=float(params.problem.depot.longitude),
                format="%.4f",
                key="param_depot_longitude",
            )

            st.markdown("**Costs**")
            st.number_input(
                "Variable Cost ($/hr)",
                value=float(params.problem.variable_cost_per_hour),
                step=1.0,
                key="param_variable_cost_per_hour",
            )
            st.number_input(
                "Light Load Penalty ($)",
                value=float(params.problem.light_load_penalty),
                step=10.0,
                key="param_light_load_penalty",
            )
            st.slider(
                "Light Load Threshold",
                0.0,
                1.0,
                value=float(params.problem.light_load_threshold),
                key="param_light_load_threshold",
            )
            st.number_input(
                "Compartment Setup Cost",
                value=float(params.problem.compartment_setup_cost),
                step=10.0,
                key="param_compartment_setup_cost",
            )

            st.markdown("**Delivery Policy**")
            st.checkbox(
                "Allow Split Stops",
                value=params.problem.allow_split_stops,
                key="param_allow_split_stops",
            )

        with st.expander("🔧 Clustering", expanded=False):
            st.selectbox(
                "Clustering Method",
                options=["combine", "minibatch_kmeans", "kmedoids", "agglomerative"],
                index=[
                    "combine",
                    "minibatch_kmeans",
                    "kmedoids",
                    "agglomerative",
                ].index(params.algorithm.clustering_method),
                key="param_clustering_method",
            )
            st.selectbox(
                "Route Time Estimation",
                options=["BHH", "TSP"],
                index=["BHH", "TSP"].index(params.algorithm.route_time_estimation),
                key="param_route_time_estimation",
            )
            st.number_input(
                "Max Cluster Depth",
                value=int(params.algorithm.clustering_max_depth),
                min_value=1,
                key="param_clustering_max_depth",
            )

        with st.expander("🔄 Post-Optimization", expanded=False):
            st.checkbox(
                "Enable Post-Optimization",
                value=params.algorithm.post_optimization,
                key="param_post_optimization",
            )
            if st.session_state.get(
                "param_post_optimization", params.algorithm.post_optimization
            ):
                st.number_input(
                    "Small Cluster Size",
                    value=int(params.algorithm.small_cluster_size),
                    key="param_small_cluster_size",
                )
                st.number_input(
                    "Nearest Merge Candidates",
                    value=int(params.algorithm.nearest_merge_candidates),
                    key="param_nearest_merge_candidates",
                )
                st.number_input(
                    "Max Improvement Iterations",
                    value=int(params.algorithm.max_improvement_iterations),
                    key="param_max_improvement_iterations",
                )

        st.divider()

        run_pressed = st.button(
            "🚀 Start Optimization",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.uploaded_data is None
                or st.session_state.optimization_running
            ),
            key="start_opt_btn",
        )

        if run_pressed:
            st.session_state.optimization_running = True
            st.session_state.optimization_results = None
            st.session_state.error_info = None
            st.rerun()

    # --- Main Content ---
    if st.session_state.optimization_running:
        # Run Logic
        temp_dir = Path(tempfile.mkdtemp(prefix="fleetmix_gui_"))
        demand_path = temp_dir / "demand.csv"
        status_file = temp_dir / "status.json"

        st.session_state.uploaded_data.to_csv(demand_path, index=False)

        # Collect latest params
        params = collect_parameters_from_ui()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"gui_run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        st.info("🔄 Optimization in progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Start Process
        if (
            "optimization_process" not in st.session_state
            or not st.session_state.optimization_process.is_alive()
        ):
            p = multiprocessing.Process(
                target=run_optimization_in_process,
                args=(str(demand_path), params, str(output_dir), str(status_file)),
            )
            p.start()
            st.session_state.optimization_process = p

        # Monitor
        process = st.session_state.optimization_process
        failed = False

        while process.is_alive():
            time.sleep(0.5)
            if status_file.exists():
                try:
                    with open(status_file) as f:
                        status = json.load(f)

                    if "error" in status:
                        st.session_state.error_info = status
                        failed = True
                        process.terminate()
                        break

                    progress = status.get("progress", 0)
                    stage = status.get("stage", "")
                    status_text.text(stage)
                    progress_bar.progress(min(progress / 100, 1.0))

                    if "solution" in status:
                        st.session_state.optimization_results = status["solution"]
                        st.session_state.optimization_results["output_dir"] = str(
                            output_dir
                        )
                        break

                except Exception:
                    pass

        process.join(timeout=2)
        if process.is_alive():
            # Force terminate if join times out (e.g. stuck threads)
            process.terminate()
            process.join()

        # Final check for solution if not yet found
        if not st.session_state.optimization_results and not failed:
            if status_file.exists():
                try:
                    with open(status_file) as f:
                        status = json.load(f)
                    if "solution" in status:
                        st.session_state.optimization_results = status["solution"]
                        st.session_state.optimization_results["output_dir"] = str(
                            output_dir
                        )
                except Exception:
                    pass

        # Cleanup
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors to avoid crashing the UI flow

        if "optimization_process" in st.session_state:
            del st.session_state.optimization_process

        st.session_state.optimization_running = False
        st.rerun()

    elif st.session_state.error_info:
        st.error(
            f"❌ Optimization Failed: {st.session_state.error_info.get('error', 'Unknown error')}"
        )
        with st.expander("Details"):
            st.code(st.session_state.error_info.get("traceback", ""))
        if st.button("Reset"):
            st.session_state.error_info = None
            st.rerun()

    elif st.session_state.optimization_results:
        res = st.session_state.optimization_results
        out_dir = Path(res["output_dir"])
        display_results(res, out_dir)

        if st.button("Start New Optimization"):
            st.session_state.optimization_results = None
            st.rerun()

    elif st.session_state.uploaded_data is not None:
        st.subheader("📊 Data Preview")
        df = st.session_state.uploaded_data
        st.dataframe(df.head(), use_container_width=True)
        st.info("Configure parameters in the sidebar and click 'Start Optimization'")

    else:
        st.info("👈 Please upload demand data to begin.")


if __name__ == "__main__":
    main()
