"""
Streamlit GUI for fleetmix optimizer.
All GUI code in one file to minimize changes.
"""

import json
import multiprocessing
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from fleetmix import api
from fleetmix.config.parameters import Parameters

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


def init_session_state():
    """Initialize session state variables."""
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "optimization_running" not in st.session_state:
        st.session_state.optimization_running = False
    if "parameters" not in st.session_state:
        st.session_state.parameters = Parameters.from_yaml()
    if "error_info" not in st.session_state:
        st.session_state.error_info = None


def convert_numpy_types(obj):
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
    demand_path: str, params_file: str, output_dir: str, status_file: str
):
    """Runs optimization in separate process to support multiprocessing."""
    try:
        # Update status
        with open(status_file, "w") as f:
            json.dump({"stage": "Initializing...", "progress": 0}, f)

        # Load parameters from YAML file
        params = Parameters.from_yaml(params_file)

        # Update status - generating clusters
        with open(status_file, "w") as f:
            json.dump({"stage": "Generating clusters...", "progress": 25}, f)

        # Set output directory
        params.results_dir = Path(output_dir)

        # Run optimization
        solution = api.optimize(
            demand=demand_path,
            config=params,
            output_dir=output_dir,
            format="excel",
            verbose=False,
        )

        # Update status - optimization complete
        with open(status_file, "w") as f:
            json.dump(
                {
                    "stage": "Optimization complete!",
                    "progress": 100,
                    "solution": convert_numpy_types(solution),
                },
                f,
            )

        return solution

    except Exception as e:
        # Write error to status file
        with open(status_file, "w") as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
        raise


def collect_parameters_from_ui() -> Parameters:
    """Build Parameters object from Streamlit widgets."""
    # Start with defaults
    params = st.session_state.parameters

    # Create a dictionary with all parameters
    params_dict = {
        "vehicles": params.vehicles,
        "goods": params.goods,
        "depot": params.depot,
        "demand_file": params.demand_file,
        "clustering": params.clustering,
        "variable_cost_per_hour": params.variable_cost_per_hour,
        "light_load_penalty": params.light_load_penalty,
        "light_load_threshold": params.light_load_threshold,
        "compartment_setup_cost": params.compartment_setup_cost,
        "format": params.format,
        "post_optimization": params.post_optimization,
        "small_cluster_size": params.small_cluster_size,
        "nearest_merge_candidates": params.nearest_merge_candidates,
        "max_improvement_iterations": params.max_improvement_iterations,
        "prune_tsp": params.prune_tsp,
        "allow_split_stops": params.allow_split_stops,
    }

    # Override with UI values stored in session state
    for key in st.session_state:
        if key.startswith("param_"):
            param_name = key[6:]  # Remove 'param_' prefix
            if param_name in params_dict:
                params_dict[param_name] = st.session_state[key]
            elif "." in param_name:
                # Handle nested parameters like clustering.method
                parts = param_name.split(".")
                if parts[0] in params_dict:
                    if isinstance(params_dict[parts[0]], dict):
                        params_dict[parts[0]][parts[1]] = st.session_state[key]

    # Convert vehicle dictionaries to VehicleSpec objects if needed
    if "vehicles" in params_dict and isinstance(params_dict["vehicles"], dict):
        vehicles_converted = {}
        for vtype, vdata in params_dict["vehicles"].items():
            if (
                isinstance(vdata, dict)
                and "capacity" in vdata
                and "fixed_cost" in vdata
            ):
                # Convert dict to VehicleSpec
                from fleetmix.core_types import VehicleSpec

                vehicles_converted[vtype] = VehicleSpec(
                    capacity=vdata["capacity"],
                    fixed_cost=vdata["fixed_cost"],
                    compartments=vdata.get("compartments", {}),
                    avg_speed=vdata.get("avg_speed", 30.0),
                    service_time=vdata.get("service_time", 25.0),
                    max_route_time=vdata.get("max_route_time", 10.0),
                    allowed_goods=vdata.get("allowed_goods"),
                    extra={
                        k: v
                        for k, v in vdata.items()
                        if k
                        not in [
                            "capacity",
                            "fixed_cost",
                            "compartments",
                            "avg_speed",
                            "service_time",
                            "max_route_time",
                            "allowed_goods",
                        ]
                    },
                )
            else:
                # Already a VehicleSpec object
                vehicles_converted[vtype] = vdata
        params_dict["vehicles"] = vehicles_converted

    # Create and return Parameters object
    new_params = Parameters(**params_dict)

    # Persist the updated parameters in the session state so that subsequent
    # reruns (triggered automatically by Streamlit) remember the user's
    # selections instead of reverting to the original defaults.  This solves
    # issues where some changes – e.g. per-vehicle average speed – appeared to
    # have no effect because the old configuration was silently restored on
    # rerun.
    st.session_state.parameters = new_params

    return new_params


def display_results(solution: dict[str, Any], output_dir: Path):
    """Display optimization results."""
    st.success("✅ Optimization completed successfully!")

    # Calculate total cost
    total_cost = (
        solution["total_fixed_cost"]
        + solution["total_variable_cost"]
        + solution["total_penalties"]
    )

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cost", f"${total_cost:,.2f}")

    with col2:
        st.metric(
            "Vehicles Used",
            solution.get("total_vehicles", len(solution["vehicles_used"])),
        )

    with col3:
        st.metric("Missing Customers", len(solution["missing_customers"]))

    with col4:
        st.metric("Solver Time", f"{solution['solver_runtime_sec']:.1f}s")

    # Cost breakdown
    st.subheader("📊 Cost Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fixed Cost", f"${solution['total_fixed_cost']:,.2f}")

    with col2:
        st.metric("Variable Cost", f"${solution['total_variable_cost']:,.2f}")

    with col3:
        st.metric("Penalties", f"${solution['total_penalties']:,.2f}")

    # Vehicle usage details
    if solution.get("vehicles_used"):
        st.subheader("🚚 Vehicle Usage")
        vehicle_list_for_df = []
        if isinstance(solution["vehicles_used"], dict):
            vehicle_list_for_df = [
                {"Vehicle Type": veh_type, "Count": veh_count}
                for veh_type, veh_count in solution["vehicles_used"].items()
            ]
        # The error message suggests the above `dict` case is what's happening.
        # If it could also be a list of dicts like [{'type': 'A', 'count':10}] or [{'vehicle_type': 'A', 'count':10}]:
        elif isinstance(solution["vehicles_used"], list):
            for item_dict in solution["vehicles_used"]:
                if isinstance(item_dict, dict):
                    # Accommodate common key names for vehicle type
                    veh_type = item_dict.get("vehicle_type", item_dict.get("type"))
                    veh_count = item_dict.get("count")
                    if veh_type is not None and veh_count is not None:
                        vehicle_list_for_df.append(
                            {"Vehicle Type": veh_type, "Count": veh_count}
                        )

        if vehicle_list_for_df:
            vehicle_df = pd.DataFrame(vehicle_list_for_df)
            st.dataframe(vehicle_df, use_container_width=True)
        elif solution[
            "vehicles_used"
        ]:  # Data was present but not in a recognized format above
            st.markdown(
                f"Vehicle usage data found, but its format was not fully recognized for table display. Data: `{solution['vehicles_used']!s}`"
            )
        # If solution['vehicles_used'] was present but empty (e.g., {} or []), no specific message needed here, subheader is enough.

    # Download section
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)

    # Find result files
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
        with open(html_files[0]) as f:
            st.components.v1.html(f.read(), height=600)


def main():
    """Main Streamlit app."""
    init_session_state()

    st.title("🚚 Fleetmix Optimizer")
    st.markdown("Optimize your fleet size and mix for heterogeneous vehicle routing")

    # --- Sidebar ---
    # Logic for run_button_pressed will be determined within the sidebar context
    _run_button_ui_element_pressed = (
        False  # Placeholder for the actual button press state
    )

    with st.sidebar:
        st.header("📋 Configuration")

        # File upload
        st.subheader("1. Upload Demand Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="Upload a CSV file with customer demand data",
        )

        if uploaded_file is not None:
            df_upload = pd.read_csv(
                uploaded_file
            )  # Use a different variable name to avoid conflict if needed
            st.session_state.uploaded_data = df_upload
            st.success(f"✓ Loaded {len(df_upload)} customers")

        # Parameters
        st.subheader("2. Configure Parameters")

        # Vehicle configuration
        with st.expander("🚚 Vehicles", expanded=False):
            st.markdown("Configure vehicle types and costs")
            st.info(
                "💡 **Tip**: Use 'Allowed Goods' to create specialized vehicles (e.g., refrigerated trucks for chilled/frozen goods only)"
            )
            vehicles = st.session_state.parameters.vehicles.copy()  # Work with a copy

            # Get available goods from parameters
            available_goods = st.session_state.parameters.goods

            for vehicle_type, vehicle_data in vehicles.items():
                st.markdown(f"**Vehicle Type {vehicle_type}**")
                col1_v, col2_v = st.columns(2)  # Unique var names for columns
                with col1_v:
                    capacity = st.number_input(
                        "Capacity",
                        min_value=100,
                        max_value=1000000,
                        value=int(
                            vehicle_data.capacity
                            if hasattr(vehicle_data, "capacity")
                            else vehicle_data["capacity"]
                        ),
                        step=100,
                        key=f"vehicle_{vehicle_type}_capacity",
                    )
                with col2_v:
                    fixed_cost = st.number_input(
                        "Fixed Cost",
                        min_value=0,
                        max_value=1000000,
                        value=int(
                            vehicle_data.fixed_cost
                            if hasattr(vehicle_data, "fixed_cost")
                            else vehicle_data["fixed_cost"]
                        ),
                        step=25,
                        key=f"vehicle_{vehicle_type}_fixed_cost",
                    )

                # Add timing parameters
                col1_t, col2_t, col3_t = st.columns(3)
                with col1_t:
                    avg_speed = st.number_input(
                        "Avg Speed (km/h)",
                        min_value=10,
                        max_value=100,
                        value=int(
                            vehicle_data.avg_speed
                            if hasattr(vehicle_data, "avg_speed")
                            else 30
                        ),
                        step=5,
                        key=f"vehicle_{vehicle_type}_avg_speed",
                    )
                with col2_t:
                    service_time = st.number_input(
                        "Service Time (min)",
                        min_value=5,
                        max_value=12000,
                        value=int(
                            vehicle_data.service_time
                            if hasattr(vehicle_data, "service_time")
                            else 25
                        ),
                        step=5,
                        key=f"vehicle_{vehicle_type}_service_time",
                    )
                with col3_t:
                    max_route_time = st.number_input(
                        "Max Route Time (h)",
                        min_value=4,
                        max_value=240,
                        value=int(
                            vehicle_data.max_route_time
                            if hasattr(vehicle_data, "max_route_time")
                            else 10
                        ),
                        step=1,
                        key=f"vehicle_{vehicle_type}_max_route_time",
                    )

                # Add allowed goods selection
                st.markdown("**Allowed Goods**")

                # Get current allowed goods for this vehicle
                current_allowed_goods = None
                if hasattr(vehicle_data, "allowed_goods"):
                    current_allowed_goods = vehicle_data.allowed_goods
                elif isinstance(vehicle_data, dict) and "allowed_goods" in vehicle_data:
                    current_allowed_goods = vehicle_data["allowed_goods"]

                # If no allowed goods specified, default to all goods
                if current_allowed_goods is None:
                    current_allowed_goods = available_goods

                allowed_goods: list[str] = st.multiselect(
                    f"Select goods that {vehicle_type} can carry",
                    options=available_goods,
                    default=current_allowed_goods,
                    key=f"vehicle_{vehicle_type}_allowed_goods",
                    help="Leave empty to allow all goods. Select specific goods to restrict this vehicle type.",
                )

                # If no goods selected, default to all goods
                if not allowed_goods:
                    allowed_goods = available_goods
                    st.info(
                        f"No goods selected - {vehicle_type} will be able to carry all goods"
                    )

                vehicles[vehicle_type] = {
                    "capacity": capacity,
                    "fixed_cost": fixed_cost,
                    "avg_speed": float(avg_speed),
                    "service_time": float(service_time),
                    "max_route_time": float(max_route_time),
                    "allowed_goods": allowed_goods,
                    # Preserve any existing compartment layout so downstream logic is unaffected
                    "compartments": (
                        vehicle_data.compartments
                        if hasattr(vehicle_data, "compartments")
                        else vehicle_data.get("compartments", {})
                    ),
                }
            st.session_state["param_vehicles"] = vehicles

        # Operations parameters
        with st.expander("⚙️ Operations", expanded=False):
            st.markdown(
                "**Note:** Vehicle-specific parameters (speed, service time, route time) are now configured per vehicle type in the config file."
            )
            st.number_input(
                "Variable Cost per Hour ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.parameters.variable_cost_per_hour),
                step=1.0,
                key="param_variable_cost_per_hour",
            )

            # Split-stop capability
            st.markdown("**Split-Stop Configuration**")
            st.checkbox(
                "Allow Split Stops",
                value=getattr(st.session_state.parameters, "allow_split_stops", False),
                key="param_allow_split_stops",
                help="Allow customers to be served by multiple vehicles. Useful for large customers with diverse goods requirements.",
            )

        # Clustering parameters
        with st.expander("🔧 Clustering", expanded=False):
            st.selectbox(
                "Clustering Method",
                options=["combine", "minibatch_kmeans", "kmedoids", "agglomerative"],
                index=[
                    "combine",
                    "minibatch_kmeans",
                    "kmedoids",
                    "agglomerative",
                ].index(st.session_state.parameters.clustering["method"]),
                key="param_clustering.method",
            )
            st.selectbox(
                "Route Time Estimation",
                options=["BHH", "TSP", "Legacy"],
                index=["BHH", "TSP", "Legacy"].index(
                    st.session_state.parameters.clustering.get(
                        "route_time_estimation", "BHH"
                    )
                ),
                key="param_clustering.route_time_estimation",
            )
            st.number_input(
                "Max Cluster Depth",
                min_value=5,
                max_value=50,
                value=int(st.session_state.parameters.clustering.get("max_depth", 20)),
                key="param_clustering.max_depth",
            )
            # No need to manually rebuild 'param_clustering' dict if keys directly map like 'param_clustering.method'

        # Cost penalties
        with st.expander("💰 Cost Penalties", expanded=False):
            st.number_input(
                "Light Load Penalty ($)",
                min_value=0.0,
                max_value=1000.0,
                value=float(st.session_state.parameters.light_load_penalty),
                step=10.0,
                key="param_light_load_penalty",
            )
            st.slider(
                "Light Load Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.parameters.light_load_threshold),
                step=0.05,
                format="%.2f",
                key="param_light_load_threshold",
                help="Threshold for considering a vehicle lightly loaded (0.2 = 20%)",
            )
            st.number_input(
                "Compartment Setup Cost ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.parameters.compartment_setup_cost),
                step=10.0,
                key="param_compartment_setup_cost",
            )

        # Post-optimization
        with st.expander("🔄 Post-Optimization", expanded=False):
            post_opt_enabled = st.checkbox(
                "Enable Post-Optimization",
                value=st.session_state.parameters.post_optimization,
                key="param_post_optimization",
            )
            if post_opt_enabled:
                st.number_input(
                    "Small Cluster Size",
                    min_value=1,
                    max_value=1000000,
                    value=int(st.session_state.parameters.small_cluster_size),
                    key="param_small_cluster_size",
                )
                st.number_input(
                    "Nearest Merge Candidates",
                    min_value=1,
                    max_value=1000000,
                    value=int(st.session_state.parameters.nearest_merge_candidates),
                    key="param_nearest_merge_candidates",
                )
                st.number_input(
                    "Max Improvement Iterations",
                    min_value=0,
                    max_value=1000000,
                    value=int(st.session_state.parameters.max_improvement_iterations),
                    key="param_max_improvement_iterations",
                )

        st.divider()
        _run_button_ui_element_pressed = st.button(
            "🚀 Start Optimization",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.uploaded_data is None
                or st.session_state.get("optimization_running", False)
            ),
            key="start_optimization_btn",
        )

    if _run_button_ui_element_pressed and st.session_state.uploaded_data is not None:
        st.session_state.optimization_running = True
        st.session_state.optimization_results = None  # Clear previous results
        st.session_state.error_info = None  # Clear previous errors
        st.rerun()

    # --- Main Content Area ---
    main_area_container = st.container()

    with main_area_container:
        if st.session_state.get("optimization_running", False):
            # Create temporary directory for this run
            temp_dir = Path(tempfile.mkdtemp(prefix="fleetmix_gui_"))
            demand_path = temp_dir / "demand.csv"
            status_file = temp_dir / "status.json"

            st.session_state.uploaded_data.to_csv(demand_path, index=False)

            params = collect_parameters_from_ui()
            params.demand_file = str(demand_path)  # Update the demand file path

            # Save parameters to a temporary YAML file for multiprocessing
            params_file = temp_dir / "params.yaml"
            params.to_yaml(params_file)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / f"gui_run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            st.info("🔄 Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing optimization process...")  # Initial message

            if (
                "optimization_process" not in st.session_state
                or not st.session_state.optimization_process.is_alive()
            ):
                st.session_state.error_info = None  # Reset error info for new run
                process = multiprocessing.Process(
                    target=run_optimization_in_process,
                    args=(
                        str(demand_path),
                        str(params_file),
                        str(output_dir),
                        str(status_file),
                    ),
                )
                process.start()
                st.session_state.optimization_process = process

            process = st.session_state.optimization_process
            optimization_failed_flag = False

            while process.is_alive():
                time.sleep(0.1)  # Check status more frequently
                if status_file.exists():
                    try:
                        with open(status_file) as f:
                            current_status = json.load(f)  # Use a different var name

                        if "error" in current_status:
                            st.session_state.error_info = {
                                "message": current_status["error"],
                                "traceback": current_status.get(
                                    "traceback", "No traceback available"
                                ),
                            }
                            optimization_failed_flag = True
                            if process.is_alive():
                                process.terminate()
                                process.join(timeout=1)
                            break

                        stage = current_status.get("stage", "Processing...")
                        progress = current_status.get("progress", 0)
                        status_text.text(stage)
                        progress_bar.progress(progress / 100)

                        if "solution" in current_status:
                            st.session_state.optimization_results = current_status[
                                "solution"
                            ]
                            if st.session_state.optimization_results is not None:
                                # Ensure output_dir is stored as string, as Path might not be ideal for session state across reruns
                                st.session_state.optimization_results["output_dir"] = (
                                    str(output_dir)
                                )
                                status_text.text(
                                    f"Solution received with keys: {list(st.session_state.optimization_results.keys()) if isinstance(st.session_state.optimization_results, dict) else 'NOT A DICT'}"
                                )

                                # If we have a complete solution (progress 100), break out of monitoring loop
                                if progress >= 100:
                                    status_text.text(
                                        "Optimization complete! Preparing results..."
                                    )
                                    break
                            else:
                                status_text.text("Solution received but it's None!")
                    except json.JSONDecodeError:
                        time.sleep(0.2)  # Wait if file is being written
                        continue
                    except Exception as e_mon:  # Catch other monitoring errors
                        st.session_state.error_info = {
                            "message": f"Error monitoring optimization: {e_mon!s}",
                            "traceback": traceback.format_exc(),
                        }
                        optimization_failed_flag = True
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1)
                        break

            if process.is_alive():  # Ensure process has finished or timeout joining
                process.join(timeout=2)

            # Final check for error or missing solution if process ended without explicit flags
            if (
                not optimization_failed_flag
                and st.session_state.get("optimization_results") is None
            ):
                final_error_message = (
                    "Optimization process completed without providing a solution."
                )
                final_traceback = "No solution found in status file or process ended unexpectedly. Check logs."
                if status_file.exists():
                    try:
                        with open(status_file) as f:
                            current_status = json.load(f)
                            if "error" in current_status:
                                final_error_message = current_status["error"]
                                final_traceback = current_status.get(
                                    "traceback", final_traceback
                                )
                    except Exception:
                        pass  # Ignore error reading final status for this specific check

                st.session_state.error_info = {
                    "message": final_error_message,
                    "traceback": final_traceback,
                }
                # optimization_failed_flag = True # Not strictly needed as error_info is set

            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if "optimization_process" in st.session_state:
                del st.session_state.optimization_process

            st.session_state.optimization_running = False  # Critical: set before rerun
            st.rerun()

        elif st.session_state.get("error_info") is not None:
            error_details = st.session_state.error_info
            st.error(f"❌ Optimization failed: {error_details['message']}")
            show_traceback = st.checkbox(
                "Show detailed error", key="show_traceback_checkbox_main"
            )
            if show_traceback:
                st.code(error_details.get("traceback", "No traceback available."))
            if st.button("Acknowledge Error and Reset", key="ack_error_btn_main"):
                st.session_state.error_info = None
                st.session_state.optimization_results = (
                    None  # Ensure results are cleared too
                )
                # optimization_running should be False already
                st.rerun()

        elif st.session_state.get("optimization_results") is not None:
            results_data = st.session_state.optimization_results
            # Ensure 'output_dir' is a Path object for display_results
            if "output_dir" in results_data and isinstance(
                results_data["output_dir"], str
            ):
                output_dir_for_display = Path(results_data["output_dir"])
                try:
                    display_results(results_data, output_dir_for_display)
                except Exception as e:
                    st.error(f"Error displaying results: {e!s}")
                    st.code(traceback.format_exc())
                    if st.button("Clear Results and Reset", key="clear_results_btn"):
                        st.session_state.optimization_results = None
                        st.rerun()
            else:
                st.error(
                    "Output directory information is missing or invalid in results."
                )
                st.write(
                    f"Results data keys: {list(results_data.keys()) if isinstance(results_data, dict) else 'NOT A DICT'}"
                )
                if st.button(
                    "Clear Results and Reset", key="clear_invalid_results_btn"
                ):
                    st.session_state.optimization_results = None
                    st.rerun()

        elif st.session_state.uploaded_data is not None:
            st.subheader("📊 Data Preview")
            df_preview = st.session_state.uploaded_data
            col1_dp, col2_dp, col3_dp = st.columns(3)  # Unique var names for columns
            with col1_dp:
                st.metric("Total Customers", len(df_preview))
            demand_cols = [
                col
                for col in df_preview.columns
                if "demand" in col.lower() or col in ["Dry", "Chilled", "Frozen"]
            ]
            if demand_cols:
                total_demand = df_preview[demand_cols].sum().sum()
                with col2_dp:
                    st.metric("Total Demand", f"{total_demand:,.0f}")
            with col3_dp:
                st.metric("Data Columns", len(df_preview.columns))
                st.dataframe(df_preview.head(10), use_container_width=True)
        else:
            st.info(
                "👈 Please upload demand data and configure parameters in the sidebar to begin."
            )


if __name__ == "__main__":
    main()
