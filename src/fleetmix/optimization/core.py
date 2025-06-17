"""
core.py

Solves the **Fleet Size-and-Mix with Heterogeneous Multi-Compartment Vehicles** optimisation
problem, corresponding to Model (2) in Section 4.3 of the research paper.

Given a pool of candidate clusters K (created in ``fleetmix.clustering`` via
:func:`generate_feasible_clusters`) and a catalogue of
vehicle configurations V, this module builds and solves an integer linear programme that
selects a subset of clusters and assigns exactly one vehicle configuration to each selected
cluster.

Mathematical formulation (paper Eq. (1)–(4))
-------------------------------------------
Objective: minimise  Σ_{v∈V} Σ_{k∈K_v} c_vk · x_vk

subject to
* Coverage – every customer appears in **at least** one chosen cluster (Eq. 2)
* Uniqueness – each cluster is selected **at most** once (Eq. 3)
* Binary decision variables x_vk and y_k (Eq. 4)

Key symbols
~~~~~~~~~~~
``x_vk``  Binary var, 1 if config *v* serves cluster *k*.
``y_k``   Binary var, 1 if cluster *k* is selected (handy for warm-starts).
``c_vk``  Total cost of dispatching configuration *v* on cluster *k* (fixed + variable).

Solver interface
----------------
• Defaults to CBC via ``pulp`` but can fall back to Gurobi/CPLEX if the corresponding environment
  variables are set (see ``utils/solver.py``).
• Post-solution **improvement phase** is optionally triggered (Section 4.4) via
  :func:`post_optimization.improve_solution`.

Typical usage
-------------
>>> from fleetmix.clustering import generate_feasible_clusters
>>> from fleetmix.optimization import optimize_fleet
>>> clusters = generate_feasible_clusters(customers, configs, params)
>>> solution = optimize_fleet(clusters, configs, customers, params)
>>> print(solution.total_cost)
"""

import sys
import time
from typing import Any

import pandas as pd  # Kept only for I/O compatibility functions
import pulp

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import (
    Cluster,
    Customer,
    CustomerBase,
    FleetmixSolution,
    VehicleConfiguration,
)
from fleetmix.post_optimization import improve_solution
from fleetmix.preprocess.demand import get_origin_id, get_subset_from_id
from fleetmix.utils.logging import Colors, FleetmixLogger, Symbols
from fleetmix.utils.solver import pick_solver

logger = FleetmixLogger.get_logger(__name__)


# Helper functions for working with List[VehicleConfiguration]
def _find_config_by_id(
    configurations: list[VehicleConfiguration], config_id: str
) -> VehicleConfiguration:
    """Find configuration by ID from list."""
    for config in configurations:
        if str(config.config_id) == str(config_id):
            return config
    raise KeyError(f"Configuration {config_id} not found")


def _create_config_lookup(
    configurations: list[VehicleConfiguration],
) -> dict[str, VehicleConfiguration]:
    """Create a dictionary lookup for configurations."""
    return {str(config.config_id): config for config in configurations}


def _configs_to_dataframe(configurations: list[VehicleConfiguration]) -> pd.DataFrame:
    """Convert configurations to DataFrame when pandas operations are needed."""
    return pd.DataFrame([config.to_dict() for config in configurations])


def optimize_fleet(
    clusters: list[Cluster],
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    parameters: Parameters,
    solver=None,
    verbose: bool = False,
    time_recorder=None,
) -> FleetmixSolution:
    """Solve the Fleet Size-and-Mix MILP.

    This is the tactical optimisation layer described in Section 4.3 of the
    paper.  It takes the candidate clusters produced during the cluster-first
    phase and decides how many vehicles of each configuration to deploy and
    which cluster each vehicle will serve.

    Args:
        clusters: List of Cluster objects from the clustering stage.
        configurations: List of vehicle configurations, each containing
            capacity, fixed cost, and compartment information.
        customers: List of Customer objects used for validation—ensures every
            customer is covered in the final solution. TODO: check if this argument is needed.
        parameters: Fully populated :class:`fleetmix.config.parameters.Parameters`
            object with cost coefficients, penalty thresholds, etc.
        solver: Optional explicit `pulp` solver instance.  If *None*,
            :func:`fleetmix.utils.solver.pick_solver` chooses CBC/Gurobi/CPLEX based
            on environment variables.
        verbose: If *True* prints solver progress to stdout.
        time_recorder: Optional TimeRecorder instance to measure post-optimization time.

    Returns:
        FleetmixSolution: A solution object with
            ``total_cost``, ``total_fixed_cost``, ``total_variable_cost``,
            ``total_penalties``, ``selected_clusters`` (DataFrame),
            ``vehicles_used`` (dict), and solver metadata.

    Example:
        >>> sol = optimize_fleet(clusters, configs, customers, params)
        >>> sol.total_cost
        10543.75

    Note:
        If ``parameters.post_optimization`` is *True* the solution may be further
        refined by :func:`fleetmix.post_optimization.improve_solution` before being
        returned.
    """
    # Call internal implementation directly with dataclass lists
    return _solve_internal(
        clusters,
        configurations,
        customers,
        parameters,
        solver,
        verbose,
        time_recorder,
    )


def _solve_internal(
    clusters: list[Cluster],
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    parameters: Parameters,
    solver=None,
    verbose: bool = False,
    time_recorder=None,
) -> FleetmixSolution:
    """Internal implementation that processes dataclass lists."""
    # Create optimization model
    model, y_vars, x_vars, c_vk = _create_model(clusters, configurations, parameters)

    # Select solver: use provided or pick based on FSM_SOLVER env
    solver = solver or pick_solver(verbose)
    logger.info(f"Using solver: {solver.name}")
    start_time = time.time()
    model.solve(solver)
    end_time = time.time()
    solver_time = end_time - start_time

    if verbose:
        print(f"Optimization completed in {solver_time:.2f} seconds.")

    # Check solution status
    if model.status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus[model.status]

        # Enhanced error message for infeasible problems
        if status_name == "Infeasible":
            # Check if any clusters have no feasible vehicles
            clusters_without_vehicles = []
            has_novehicle_vars = False

            for cluster in clusters:
                cluster_id = cluster.cluster_id
                has_feasible_vehicle = False

                for config in configurations:
                    # Check if vehicle can serve cluster
                    total_demand = sum(cluster.total_demand.values())
                    goods_required = set(
                        g for g in parameters.goods if cluster.total_demand.get(g, 0) > 0
                    )

                    if total_demand <= config.capacity:
                        # Check goods compatibility
                        if all(config[g] == 1 for g in goods_required):
                            has_feasible_vehicle = True
                            break

                if not has_feasible_vehicle:
                    clusters_without_vehicles.append(cluster_id)

            # Check if we have NoVehicle variables in the x_vars
            has_novehicle_vars = any(v == "NoVehicle" for v, k in x_vars.keys())

            # If we have NoVehicle variables and infeasible clusters, this should be
            # treated as "Not Solved" rather than "Infeasible" in traditional mode
            if (
                has_novehicle_vars
                and clusters_without_vehicles
                and not parameters.allow_split_stops
            ):
                error_msg = "Optimization failed with status: Not Solved"
                raise RuntimeError(error_msg)

            error_msg = "Optimization problem is infeasible!"

            if clusters_without_vehicles:
                error_msg += (
                    f"\nClusters without feasible vehicles: {clusters_without_vehicles}"
                )
                error_msg += "\nPossible causes:"
                error_msg += "\n- Vehicle capacities are too small for cluster demands"
                error_msg += "\n- No vehicles have the right compartment mix"
                error_msg += "\n- Consider adding larger vehicles or more compartment configurations"
            else:
                error_msg += (
                    "\nAll clusters have feasible vehicles, but constraints conflict."
                )
                error_msg += "\nPossible causes:"
                error_msg += "\n- Not enough vehicles (check max_vehicles parameter)"
                error_msg += "\n- Customer coverage constraints cannot be satisfied"
                error_msg += "\n- Try relaxing penalties or adding more vehicle types"

            raise ValueError(error_msg)
        else:
            error_msg = f"Optimization failed with status: {status_name}"
            raise RuntimeError(error_msg)

    # Extract and validate solution
    selected_clusters = _extract_solution(clusters, y_vars, x_vars)
    missing_customers = _validate_solution(
        selected_clusters, customers, configurations, parameters
    )

    # Add goods columns from configurations before calculating statistics
    config_lookup = _create_config_lookup(configurations)
    for good in parameters.goods:
        selected_clusters[good] = selected_clusters["Config_ID"].map(
            lambda x: config_lookup[str(x)][good]
        )

    # Calculate statistics using the actual optimization costs
    solution = _calculate_solution_statistics(
        selected_clusters, configurations, parameters, model, x_vars, c_vk
    )

    # Add additional solution data by setting attributes directly
    solution.missing_customers = missing_customers
    solution.solver_name = model.solver.name
    solution.solver_status = pulp.LpStatus[model.status]
    solution.solver_runtime_sec = solver_time

    # Improvement phase
    post_optimization_time = None
    if parameters.post_optimization:
        if time_recorder:
            with time_recorder.measure("fsm_post_optimization"):
                post_start = time.time()
                solution = improve_solution(
                    solution, configurations, customers, parameters
                )
                post_end = time.time()
                post_optimization_time = post_end - post_start
        else:
            post_start = time.time()
            solution = improve_solution(
                solution, configurations, customers, parameters
            )
            post_end = time.time()
            post_optimization_time = post_end - post_start

    # Record post-optimization runtime
    solution.post_optimization_runtime_sec = post_optimization_time

    return solution


def _create_model(
    clusters: list[Cluster],
    configurations: list[VehicleConfiguration],
    parameters: Parameters,
) -> tuple[
    pulp.LpProblem,
    dict[Any, pulp.LpVariable],
    dict[tuple[Any, Any], pulp.LpVariable],
    dict[tuple[Any, Any], float],
]:
    """
    Create the optimization model M aligning with the mathematical formulation.
    """
    import pulp

    # Create the optimization model
    model = pulp.LpProblem("FSM-MCV_Model2", pulp.LpMinimize)

    # Sets
    N = set()  # Customers
    for cluster in clusters:
        N.update(cluster.customers)
    K = set(cluster.cluster_id for cluster in clusters)  # Clusters

    # Initialize decision variables dictionaries
    x_vars = {}
    y_vars = {}
    c_vk = {}

    # K_i: clusters containing customer i
    K_i: dict[Any, set[Any]] = {i: set() for i in N}
    for cluster in clusters:
        cluster_id = cluster.cluster_id
        for customer_id in cluster.customers:
            K_i[customer_id].add(cluster_id)

    # V_k: vehicle configurations that can serve cluster k
    V_k: dict[Any, set[Any]] = {}
    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    
    for k in K:
        V_k[k] = set()
        cluster = cluster_lookup[k]
        cluster_goods_required = set(
            g for g in parameters.goods if cluster.total_demand.get(g, 0) > 0
        )
        q_k = sum(cluster.total_demand.values())

        for config in configurations:
            v = config.config_id
            # Check capacity
            if q_k > config.capacity:
                continue  # Vehicle cannot serve this cluster

            # Check product compatibility
            compatible = all(config[g] == 1 for g in cluster_goods_required)

            if compatible:
                V_k[k].add(v)

        # If V_k[k] is empty, handle accordingly
        if not V_k[k]:
            logger.debug(f"Cluster {k} cannot be served by any vehicle configuration.")
            # Force y_k to 0 (cluster cannot be selected)
            V_k[k].add("NoVehicle")  # Placeholder
            x_vars["NoVehicle", k] = pulp.LpVariable(f"x_NoVehicle_{k}", cat="Binary")
            model += x_vars["NoVehicle", k] == 0
            c_vk["NoVehicle", k] = 0.0  # Cost is zero as it's not selected

    # Create remaining decision variables
    for k in K:
        y_vars[k] = pulp.LpVariable(f"y_{k}", cat="Binary")
        for v in V_k[k]:
            if (v, k) not in x_vars:  # Only create if not already created
                x_vars[v, k] = pulp.LpVariable(f"x_{v}_{k}", cat="Binary")

    # Parameters
    for k in K:
        cluster = cluster_lookup[k]
        for v in V_k[k]:
            if v != "NoVehicle":
                config = _find_config_by_id(configurations, v)
                # Calculate load percentage
                total_demand = sum(cluster.total_demand.get(g, 0) for g in parameters.goods)
                capacity = float(config.capacity)
                load_percentage = total_demand / capacity

                # Apply fixed penalty if under threshold
                penalty_amount = (
                    float(parameters.light_load_penalty)
                    if load_percentage < parameters.light_load_threshold
                    else 0.0
                )
                base_cost = _calculate_cluster_cost(
                    cluster=cluster, config=config, parameters=parameters
                )

                c_vk[v, k] = float(base_cost + penalty_amount)
                logger.debug(
                    f"Cluster {k}, vehicle {v}: Load Percentage = {load_percentage:.2f}, "
                    f"Penalty = {penalty_amount}"
                )
            else:
                c_vk[v, k] = 0.0  # Cost is zero for placeholder

    # Objective Function
    model += (
        pulp.lpSum(c_vk[v, k] * x_vars[v, k] for k in K for v in V_k[k]),
        "Total_Cost",
    )

    # Constraints

    # 1. Customer Allocation Constraint (Exact Assignment or Split-Stop Exclusivity)
    if parameters.allow_split_stops:
        # Build mapping tables for split-stop constraints
        origin_id = {customer_id: get_origin_id(customer_id) for customer_id in N}
        subset = {customer_id: get_subset_from_id(customer_id) for customer_id in N}

        # Get all physical customers and their goods
        physical_customers = set(origin_id.values())
        goods_by_physical: dict[str, set[str]] = {}
        for physical_customer in physical_customers:
            goods_by_physical[physical_customer] = set()
            for customer_id in N:
                if origin_id[customer_id] == physical_customer:
                    goods_by_physical[physical_customer].update(subset[customer_id])

        # Exclusivity constraints: each physical customer's each good must be served exactly once
        for physical_customer in physical_customers:
            for good in goods_by_physical[physical_customer]:
                model += (
                    pulp.lpSum(
                        x_vars[v, k]
                        for customer_id in N
                        if origin_id[customer_id] == physical_customer
                        and good in subset[customer_id]
                        for k in K_i[customer_id]
                        for v in V_k[k]
                        if v != "NoVehicle"
                    )
                    == 1,
                    f"Cover_{physical_customer}_{good}",
                )

        logger.info(
            f"Added split-stop exclusivity constraints for {len(physical_customers)} physical customers"
        )
    else:
        # Standard customer coverage constraint: each customer served exactly once
        for i in N:
            model += (
                pulp.lpSum(
                    x_vars[v, k] for k in K_i[i] for v in V_k[k] if v != "NoVehicle"
                )
                == 1,
                f"Customer_Coverage_{i}",
            )

    # 2. Vehicle Configuration Assignment Constraint
    for k in K:
        model += (
            (pulp.lpSum(x_vars[v, k] for v in V_k[k]) == y_vars[k]),
            f"Vehicle_Assignment_{k}",
        )

    # 3. Unserviceable Clusters Constraint
    for k in K:
        if "NoVehicle" in V_k[k]:
            model += y_vars[k] == 0, f"Unserviceable_Cluster_{k}"

    return model, y_vars, x_vars, c_vk


def _extract_solution(
    clusters: list[Cluster], y_vars: dict, x_vars: dict
) -> pd.DataFrame:
    """Extract the selected clusters and their assigned configurations. 
    For UI / export only; core algorithms run on dataclass lists."""
    selected_cluster_ids = [
        cid for cid, var in y_vars.items() if var.varValue and var.varValue > 0.5
    ]

    cluster_config_map = {}
    for (v, k), var in x_vars.items():
        if var.varValue and var.varValue > 0.5 and k in selected_cluster_ids:
            cluster_config_map[k] = v

    # Get selected clusters from list
    selected_clusters = [
        cluster for cluster in clusters if cluster.cluster_id in selected_cluster_ids
    ]

    # Update Config_ID and convert to DataFrame for compatibility with downstream code
    selected_clusters_df = Cluster.to_dataframe(selected_clusters)
    
    # Handle empty DataFrame case
    if len(selected_clusters_df) > 0:
        selected_clusters_df["Config_ID"] = selected_clusters_df["Cluster_ID"].map(
            cluster_config_map
        )
    else:
        # Ensure Config_ID column exists even for empty DataFrame
        selected_clusters_df["Config_ID"] = pd.Series(dtype=object)

    return selected_clusters_df


def _validate_solution(
    selected_clusters: pd.DataFrame,
    customers: list[CustomerBase],
    configurations: list[VehicleConfiguration],
    parameters: Parameters,
) -> set:
    """
    Validate that all customers are served in the solution.
    """
    # In split-stop mode, MILP ensures per-good coverage; skip validation of pseudo-customers
    if parameters.allow_split_stops:
        return set()
    all_customers_set = set(customer.customer_id for customer in customers)
    served_customers = set()
    for _, cluster in selected_clusters.iterrows():
        served_customers.update(cluster["Customers"])

    missing_customers = all_customers_set - served_customers
    if missing_customers:
        logger.warning(
            f"\n{Symbols.CROSS} {len(missing_customers)} customers are not served!"
        )

        # Print unserved customer demands
        unserved_customers = [c for c in customers if c.customer_id in missing_customers]
        logger.warning(
            f"{Colors.YELLOW}→ Unserved Customers:{Colors.RESET}\n"
            f"{Colors.GRAY}  Customer ID  Dry  Chilled  Frozen{Colors.RESET}"
        )

        for customer in unserved_customers:
            logger.warning(
                f"{Colors.YELLOW}  {customer.customer_id:>10}  "
                f"{customer.demands.get('Dry', 0):>3.0f}  "
                f"{customer.demands.get('Chilled', 0):>7.0f}  "
                f"{customer.demands.get('Frozen', 0):>6.0f}{Colors.RESET}"
            )

    return missing_customers


def _calculate_solution_statistics(
    selected_clusters: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    parameters: Parameters,
    model: pulp.LpProblem,
    x_vars: dict,
    c_vk: dict,
) -> FleetmixSolution:
    """Calculate solution statistics using the optimization results."""

    # Get selected assignments and their actual costs from the optimization
    selected_assignments = {
        (v, k): c_vk[(v, k)] for (v, k), var in x_vars.items() if var.varValue == 1
    }

    # Calculate compartment penalties
    total_compartment_penalties = sum(
        parameters.compartment_setup_cost
        * (sum(1 for g in parameters.goods if row[g] == 1) - 1)
        for _, row in selected_clusters.iterrows()
        if sum(1 for g in parameters.goods if row[g] == 1) > 1
    )

    # Get vehicle statistics and fixed costs
    # Drop potentially clashing columns from selected_clusters before merging
    # to ensure columns from configurations_df are used without suffixes.
    potential_clash_cols = ["Fixed_Cost", "Vehicle_Type", "Capacity"]
    cols_to_drop_from_selected = [
        col for col in potential_clash_cols if col in selected_clusters.columns
    ]
    if cols_to_drop_from_selected:
        selected_clusters = selected_clusters.drop(columns=cols_to_drop_from_selected)

    # Select only necessary columns from configurations_df to avoid merge conflicts with goods columns
    # already present in selected_clusters (this part is already good)
    cols_to_merge_from_config = ["Config_ID", "Fixed_Cost", "Vehicle_Type", "Capacity"]
    config_subset_for_merge = _configs_to_dataframe(configurations)[
        cols_to_merge_from_config
    ]

    selected_clusters = selected_clusters.merge(
        config_subset_for_merge, on="Config_ID", how="left"
    )

    # Calculate base costs (without penalties)
    if len(selected_clusters) > 0:
        total_fixed_cost = selected_clusters["Fixed_Cost"].sum()
        total_variable_cost = (
            selected_clusters["Route_Time"] * parameters.variable_cost_per_hour
        ).sum()
    else:
        total_fixed_cost = 0.0
        total_variable_cost = 0.0

    # Total cost from optimization
    total_cost = sum(selected_assignments.values())

    # Light load penalties are the remaining difference
    total_light_load_penalties = total_cost - (
        total_fixed_cost + total_variable_cost + total_compartment_penalties
    )

    # Total penalties
    total_penalties = total_light_load_penalties + total_compartment_penalties

    return FleetmixSolution(
        selected_clusters=selected_clusters,
        total_fixed_cost=total_fixed_cost,
        total_variable_cost=total_variable_cost,
        total_light_load_penalties=total_light_load_penalties,
        total_compartment_penalties=total_compartment_penalties,
        total_penalties=total_penalties,
        total_cost=total_cost,
        vehicles_used=selected_clusters["Vehicle_Type"]
        .value_counts()
        .sort_index()
        .to_dict() if len(selected_clusters) > 0 else {},
        total_vehicles=len(selected_clusters),
    )


def _calculate_cluster_cost(
    cluster: Cluster, config: VehicleConfiguration, parameters: Parameters
) -> float:
    """
    Calculate the base cost for serving a cluster with a vehicle configuration.
    Includes:
    - Fixed cost
    - Variable cost (time-based)
    - Compartment setup cost

    Note: Light load penalties are handled separately in the model creation.
    Args:
        cluster: The cluster data as a Cluster object.
        config: The vehicle configuration data as a VehicleConfiguration object.
        parameters: Parameters object containing optimization parameters.

    Returns:
        Base cost of serving the cluster with the given vehicle configuration.
    """
    # Base costs
    fixed_cost = float(config.fixed_cost)
    route_time = float(cluster.route_time)
    variable_cost = float(parameters.variable_cost_per_hour) * route_time

    # Compartment setup cost
    num_compartments = sum(1 for g in parameters.goods if config[g])
    compartment_cost = 0.0
    if num_compartments > 1:
        compartment_cost = float(parameters.compartment_setup_cost) * (
            num_compartments - 1
        )

    # Total cost
    total_cost = fixed_cost + variable_cost + compartment_cost

    return float(total_cost)
