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
• Post-solution **improvement phase** (Section 4.4) can be applied separately via
  :func:`post_optimization.improve_solution`.

Typical usage
-------------
>>> from fleetmix.clustering import generate_feasible_clusters
>>> from fleetmix.optimization import optimize_fleet
>>> clusters = generate_feasible_clusters(customers, configs, params)
>>> solution = optimize_fleet(clusters, configs, customers, params)
>>> print(solution.total_cost)
"""

import os
import sys
import time
from typing import Any

import pandas as pd
import pulp

from fleetmix.core_types import (
    Cluster,
    Customer,
    CustomerBase,
    FleetmixSolution,
    VehicleConfiguration,
)
from fleetmix.preprocess.demand import get_origin_id
from fleetmix.utils.cluster_conversion import dataframe_to_clusters
from fleetmix.utils.debug import ModelDebugger
from fleetmix.utils.logging import Colors, FleetmixLogger, Symbols
from fleetmix.utils.solver import pick_solver
from fleetmix.config.params import FleetmixParams

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
    parameters: FleetmixParams,
    solver=None,
    time_recorder=None,
    warm_start_solution: FleetmixSolution | None = None,
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
            customer is covered in the final solution.
        parameters: Fully populated :class:`fleetmix.config.parameters.Parameters`
            object with cost coefficients, penalty thresholds, etc.
        solver: Optional explicit `pulp` solver instance.  If *None*,
            :func:`fleetmix.utils.solver.pick_solver` chooses CBC/Gurobi/CPLEX based
            on environment variables.
        verbose: If *True* prints solver progress to stdout.
        time_recorder: Optional TimeRecorder instance to measure post-optimization time.
        warm_start_solution: Optional FleetmixSolution from Phase 1 to use as warm start.

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
        This function only performs the core MILP optimization. For post-optimization
        improvement, call :func:`fleetmix.post_optimization.improve_solution` separately
        on the returned solution.
    """
    # Convert to DataFrames for internal processing
    clusters_df = Cluster.to_dataframe(clusters)
    customers_df = Customer.to_dataframe(customers)

    # TODO: solver es parte de fleetmixparams.runtime, no usar pick_solver
    # Call internal implementation
    return _solve_internal(
        clusters_df,
        configurations,
        customers_df,
        parameters,
        solver,
        time_recorder,
        warm_start_solution,
    )


def _solve_internal(
    clusters_df: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    customers_df: pd.DataFrame,
    parameters: FleetmixParams,
    solver=None,
    time_recorder=None,
    warm_start_solution: FleetmixSolution | None = None,
) -> FleetmixSolution:
    """Internal implementation that processes DataFrames."""
    # Create optimization model
    model, y_vars, x_vars, c_vk = _create_model(
        clusters_df,
        configurations,
        customers_df,
        parameters,
        warm_start_solution,
    )

    # Handle empty model case (no clusters)
    if not y_vars and not x_vars:
        logger.warning("No feasible clusters - returning empty solution")
        # Return empty solution
        empty_solution = FleetmixSolution(
            total_cost=0.0,
            total_fixed_cost=0.0,
            total_variable_cost=0.0,
            total_penalties=0.0,
            selected_clusters=[],
            vehicles_used={},
            total_vehicles=0,
            missing_customers=set(customers_df["Customer_ID"]),
            solver_name="None",
            solver_status="No clusters to optimize",
            solver_runtime_sec=0.0,
            time_measurements=None,
        )
        return empty_solution

    # Select solver: use provided or pick based on FSM_SOLVER env
    solver = solver or pick_solver(parameters.runtime)
    # TODO: check why solver is passed as argument
    logger.info(f"Using solver: {solver.name}")
    start_time = time.time()
    model.solve(solver)
    end_time = time.time()
    solver_time = end_time - start_time

    if parameters.runtime.verbose:
        print(f"Optimization completed in {solver_time:.2f} seconds.")

    # Dump model artifacts if debugging is enabled
    ModelDebugger.dump(model, "fsm_model")

    # Check solution status
    if model.status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus[model.status]
        is_infeasible = status_name in ["Infeasible", "Not Solved"]

        if is_infeasible:
            error_msg = f"Optimization failed with status: {status_name}"
            if status_name == "Infeasible":
                raise ValueError(error_msg)
            else:
                raise RuntimeError(error_msg)

    # Extract and validate solution
    selected_clusters = _extract_solution(clusters_df, y_vars, x_vars)
    missing_customers = _validate_solution(
        selected_clusters, customers_df, configurations, parameters
    )

    # Add goods columns from configurations before calculating statistics
    config_lookup = _create_config_lookup(configurations)
    for good in parameters.problem.goods:
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

    return solution


def _create_model(
    clusters_df: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    customers_df: pd.DataFrame,
    parameters: FleetmixParams,
    warm_start_solution: FleetmixSolution | None = None,
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

    # Handle empty clusters case
    if clusters_df.empty:
        logger.warning("No clusters provided to optimization - creating empty model")
        # Return empty model with no variables
        return model, {}, {}, {}

    # Sets
    N = set(clusters_df["Customers"].explode().unique())  # Customers
    K = set(clusters_df["Cluster_ID"])  # Clusters

    # Initialize decision variables dictionaries
    x_vars = {}
    y_vars = {}
    c_vk = {}

    # K_i: clusters containing customer i
    K_i = {
        i: set(
            clusters_df[clusters_df["Customers"].apply(lambda x: i in x)]["Cluster_ID"]
        )
        for i in N
    }

    # V_k: vehicle configurations that can serve cluster k
    V_k: dict[Any, set[Any]] = {}
    for k in K:
        V_k[k] = set()
        cluster = clusters_df.loc[clusters_df["Cluster_ID"] == k].iloc[0]
        cluster_goods_required = set(
            g for g in parameters.problem.goods if cluster["Total_Demand"][g] > 0
        )
        q_k = sum(cluster["Total_Demand"].values())

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
        cluster = clusters_df.loc[clusters_df["Cluster_ID"] == k].iloc[0]
        for v in V_k[k]:
            if v != "NoVehicle":
                config = _find_config_by_id(configurations, v)
                # Calculate load percentage
                total_demand = sum(
                    cluster["Total_Demand"][g] for g in parameters.problem.goods
                )
                capacity = float(config.capacity)
                load_percentage = total_demand / capacity

                # Apply fixed penalty if under threshold
                penalty_amount = (
                    float(parameters.problem.light_load_penalty)
                    if load_percentage < parameters.problem.light_load_threshold
                    else 0.0
                )
                base_cost = _calculate_cluster_cost(
                    cluster=cluster, config=config, parameters=parameters
                )

                c_vk[v, k] = float(base_cost + penalty_amount)
            else:
                c_vk[v, k] = 0.0  # Cost is zero for placeholder

    # Objective Function
    model += (
        pulp.lpSum(c_vk[v, k] * x_vars[v, k] for k in K for v in V_k[k]),
        "Total_Cost",
    )

    # Constraints

    # 1. Customer Allocation Constraint (Exact Assignment or Split-Stop Exclusivity)
    if parameters.problem.allow_split_stops:
        # Build mapping tables for split-stop constraints
        # First, convert customer IDs to Customer objects to use proper methods
        customers = Customer.from_dataframe(customers_df)
        customer_objects = {c.customer_id: c for c in customers}

        origin_id = {}
        subset = {}
        for customer_id in N:
            if customer_id in customer_objects:
                customer_obj = customer_objects[customer_id]
                origin_id[customer_id] = customer_obj.get_origin_id()
                subset[customer_id] = customer_obj.get_goods_subset()
            else:
                # Fallback for any missing customers
                origin_id[customer_id] = get_origin_id(customer_id)
                subset[customer_id] = tuple()  # Empty for missing customers

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
                # Deduplicate cluster IDs to ensure each x_{v,k} appears at most once
                clusters_covering = {
                    k
                    for customer_id in N
                    if origin_id[customer_id] == physical_customer
                    and good in subset[customer_id]
                    for k in K_i[customer_id]
                }

                model += (
                    pulp.lpSum(
                        x_vars[v, k]
                        for k in clusters_covering
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

    # ------------------------------------------------------------------
    # Warm-start: Apply warm start from Phase 1 baseline solution or
    # existing baseline cluster warm start logic
    # ------------------------------------------------------------------

    # TODO: check warm-start logic, flags, etc.
    if parameters.problem.allow_split_stops:
        # Identify baseline clusters (those without "::" pseudo-customers)
        baseline_cluster_ids = []
        for k in K:
            cluster = clusters_df.loc[clusters_df["Cluster_ID"] == k].iloc[0]
            customers_in_cluster = cluster["Customers"]
            has_pseudo = any("::" in str(c) for c in customers_in_cluster)
            if not has_pseudo:
                baseline_cluster_ids.append(k)

        if warm_start_solution and os.getenv("FLEETMIX_WARMSTART", "1") == "1":
            # Phase 2: Use Phase 1 solution as warm start by mapping to baseline clusters
            logger.info(
                f"Using Phase 1 solution as warm start with {len(baseline_cluster_ids)} baseline clusters"
            )

            # Map Phase 1 clusters to Phase 2 baseline clusters by customer overlap
            phase1_clusters = warm_start_solution.selected_clusters
            warm_start_assignments = []

            for phase1_cluster in phase1_clusters:
                phase1_customers = set(phase1_cluster.customers)
                best_match_k = None
                best_overlap = 0

                # Find baseline cluster with maximum customer overlap
                for k in baseline_cluster_ids:
                    cluster = clusters_df.loc[clusters_df["Cluster_ID"] == k].iloc[0]
                    k_customers = set(cluster["Customers"])
                    overlap = len(phase1_customers.intersection(k_customers))

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_k = k

                if best_match_k and best_overlap > 0:
                    # Find best vehicle config for this cluster
                    best_v = None
                    best_cost = float("inf")
                    for v in V_k[best_match_k]:
                        if (
                            v != "NoVehicle"
                            and c_vk.get((v, best_match_k), float("inf")) < best_cost
                        ):
                            best_cost = c_vk[v, best_match_k]
                            best_v = v

                    if best_v:
                        warm_start_assignments.append((best_v, best_match_k))

            # Apply warm start values
            for v, k in warm_start_assignments:
                y_vars[k].setInitialValue(1)
                x_vars[v, k].setInitialValue(1)

            logger.info(
                f"Applied warm start to {len(warm_start_assignments)} baseline clusters"
            )

        elif baseline_cluster_ids and os.getenv("FLEETMIX_WARMSTART", "1") == "1":
            # Fallback: Use existing baseline warm start logic
            logger.info(
                f"Warm-starting with {len(baseline_cluster_ids)} baseline clusters"
            )

            for k in baseline_cluster_ids:
                y_vars[k].setInitialValue(1)
                best_v = None
                best_cost = float("inf")
                for v in V_k[k]:
                    if v != "NoVehicle" and c_vk.get((v, k), float("inf")) < best_cost:
                        best_cost = c_vk[v, k]
                        best_v = v
                if best_v:
                    x_vars[best_v, k].setInitialValue(1)

    return model, y_vars, x_vars, c_vk


def _extract_solution(
    clusters_df: pd.DataFrame, y_vars: dict, x_vars: dict
) -> pd.DataFrame:
    """Extract the selected clusters and their assigned configurations."""
    selected_cluster_ids = [
        cid for cid, var in y_vars.items() if var.varValue and var.varValue > 0.5
    ]

    cluster_config_map = {}
    for (v, k), var in x_vars.items():
        if var.varValue and var.varValue > 0.5 and k in selected_cluster_ids:
            cluster_config_map[k] = v

    # Get selected clusters with ALL columns from input DataFrame
    # This preserves the goods columns that were set during merging
    selected_clusters = clusters_df[
        clusters_df["Cluster_ID"].isin(selected_cluster_ids)
    ].copy()

    # Update Config_ID while keeping existing columns
    selected_clusters["Config_ID"] = selected_clusters["Cluster_ID"].map(
        cluster_config_map
    )

    return selected_clusters


def _validate_solution(
    selected_clusters: pd.DataFrame,
    customers_df: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    parameters: FleetmixParams,
) -> set:
    """
    Validate that all customers are served in the solution.
    """
    # In split-stop mode, MILP ensures per-good coverage; skip validation of pseudo-customers
    if parameters.problem.allow_split_stops:
        return set()
    all_customers_set = set(customers_df["Customer_ID"])
    served_customers = set()
    for _, cluster in selected_clusters.iterrows():
        served_customers.update(cluster["Customers"])

    missing_customers = all_customers_set - served_customers
    if missing_customers:
        logger.warning(
            f"\n{Symbols.CROSS} {len(missing_customers)} customers are not served!"
        )

        # Print unserved customer demands
        unserved = customers_df[customers_df["Customer_ID"].isin(missing_customers)]
        logger.warning(
            f"{Colors.YELLOW}→ Unserved Customers:{Colors.RESET}\n"
            f"{Colors.GRAY}  Customer ID  Dry  Chilled  Frozen{Colors.RESET}"
        )

        for _, customer in unserved.iterrows():
            logger.warning(
                f"{Colors.YELLOW}  {customer['Customer_ID']:>10}  "
                f"{customer['Dry_Demand']:>3.0f}  "
                f"{customer['Chilled_Demand']:>7.0f}  "
                f"{customer['Frozen_Demand']:>6.0f}{Colors.RESET}"
            )

    return missing_customers


def _calculate_solution_statistics(
    selected_clusters: pd.DataFrame,
    configurations: list[VehicleConfiguration],
    parameters: FleetmixParams,
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
        parameters.problem.compartment_setup_cost
        * (sum(1 for g in parameters.problem.goods if row[g] == 1) - 1)
        for _, row in selected_clusters.iterrows()
        if sum(1 for g in parameters.problem.goods if row[g] == 1) > 1
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
    total_fixed_cost = selected_clusters["Fixed_Cost"].sum()
    total_variable_cost = (
        selected_clusters["Route_Time"] * parameters.problem.variable_cost_per_hour
    ).sum()

    # Total cost from optimization
    total_cost = sum(selected_assignments.values())

    # Light load penalties are the remaining difference
    total_light_load_penalties = total_cost - (
        total_fixed_cost + total_variable_cost + total_compartment_penalties
    )

    # Total penalties
    total_penalties = total_light_load_penalties + total_compartment_penalties

    # Convert DataFrame to list[Cluster] before returning
    selected_clusters_list = dataframe_to_clusters(selected_clusters)

    return FleetmixSolution(
        selected_clusters=selected_clusters_list,
        total_fixed_cost=total_fixed_cost,
        total_variable_cost=total_variable_cost,
        total_light_load_penalties=total_light_load_penalties,
        total_compartment_penalties=total_compartment_penalties,
        total_penalties=total_penalties,
        total_cost=total_cost,
        vehicles_used=selected_clusters["Vehicle_Type"]
        .value_counts()
        .sort_index()
        .to_dict(),
        total_vehicles=len(selected_clusters),
    )


def _calculate_cluster_cost(
    cluster: pd.Series, config: VehicleConfiguration, parameters: FleetmixParams
) -> float:
    """
    Calculate the base cost for serving a cluster with a vehicle configuration.
    Includes:
    - Fixed cost
    - Variable cost (time-based)
    - Compartment setup cost

    Note: Light load penalties are handled separately in the model creation.
    Args:
        cluster: The cluster data as a Pandas Series.
        config: The vehicle configuration data as a VehicleConfiguration object.
        parameters: Parameters object containing optimization parameters.

    Returns:
        Base cost of serving the cluster with the given vehicle configuration.
    """
    # Base costs
    fixed_cost = float(config.fixed_cost)
    route_time = float(cluster["Route_Time"])
    variable_cost = float(parameters.problem.variable_cost_per_hour) * route_time

    # Compartment setup cost
    num_compartments = sum(1 for g in parameters.problem.goods if config[g])
    compartment_cost = 0.0
    if num_compartments > 1:
        compartment_cost = float(parameters.problem.compartment_setup_cost) * (
            num_compartments - 1
        )

    # Total cost
    total_cost = fixed_cost + variable_cost + compartment_cost

    return float(total_cost)
