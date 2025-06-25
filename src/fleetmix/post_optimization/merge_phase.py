"""
merge_phase.py

Implements the **improvement phase** (Section 4.4 of the paper) that iteratively tries to merge
small, neighbouring clusters after the core FSM model has been solved.

Rationale
~~~~~~~~~
The MILP in ``fleetmix.optimization.solve_fsm_problem`` chooses from a *fixed* pool of clusters.  Once an
initial solution is available, additional cost savings can sometimes be obtained by *merging* two
clusters and serving the combined demand with a larger vehicle—provided capacity and route‐time
constraints remain feasible.

Algorithm outline
-----------------
1. Identify "small" clusters (≤ ``params.small_cluster_size`` customers).
2. For each small cluster *s* find the nearest candidate cluster *t* that is:
   • compatible in product mix,
   • within capacity, and
   • likely feasible on route‐time (quick lower bound check).
3. Evaluate the merged cluster precisely using :func:`utils.route_time.estimate_route_time` to
   verify that the merged cluster's estimated route time does not exceed the vehicle's maximum route time.
4. Collect all feasible merges, append them to the cluster pool, and re‐optimise the MILP **without
   triggering a recursive improvement**.
5. Repeat until no further cost reduction is achieved or the iteration cap
   ``params.max_improvement_iterations`` is reached.

Caching & performance
---------------------
Route‐time calculations for the same customer sets are memoised in the module‐level dict
``_merged_route_time_cache``.

Outcome
-------
Returns the *best* improved solution dictionary, identical in structure to the one produced by
``fleetmix.optimization.solve_fsm_problem`` but with potentially lower total cost.
"""

from dataclasses import replace

import pandas as pd

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import (
    Customer,
    CustomerBase,
    FleetmixSolution,
    VehicleConfiguration,
)
from fleetmix.merging.core import (
    _create_config_lookup,
    generate_merge_phase_clusters,
)
from fleetmix.utils.logging import FleetmixLogger, Symbols
from fleetmix.utils.solver import pick_solver

logger = FleetmixLogger.get_logger(__name__)


def improve_solution(
    initial_solution: FleetmixSolution,
    configurations: list[VehicleConfiguration],
    customers: list[CustomerBase],
    params: Parameters,
) -> FleetmixSolution:
    """Iteratively merge small clusters to lower total cost.

    Implements the *improvement phase* described in Section 4.4.  Starting from
    an existing solution dictionary, the algorithm searches for pairs of
    "small" clusters that can be feasibly served together by the same vehicle
    configuration and whose merge reduces the overall objective value.

    Args:
        initial_solution: Solution object returned by
            :func:`fleetmix.optimization.solve_fsm_problem`.
        configurations: List of vehicle configurations, each containing
            capacity, fixed cost, and compartment information.
        customers: List of CustomerBase objects; required for route-time
            recalculation and centroid updates when evaluating merges.
        params: Parameter object controlling thresholds such as
            ``small_cluster_size``, ``max_improvement_iterations``, etc.

    Returns:
        FleetmixSolution: Improved solution object (same schema as *initial_solution*).
        If no improving merge is found the original object is returned.

    Example:
        >>> improved = improve_solution(sol, configs, customers, params)
        >>> improved.total_cost <= sol.total_cost
        True
    """
    # Convert to DataFrame for internal processing
    customers_df = Customer.to_dataframe(customers)

    # Call internal implementation
    return _improve_internal(initial_solution, configurations, customers_df, params)


def _improve_internal(
    initial_solution: FleetmixSolution,
    configurations: list[VehicleConfiguration],
    customers_df: pd.DataFrame,
    params: Parameters,
) -> FleetmixSolution:
    """Internal implementation that processes DataFrames."""

    best_solution = initial_solution
    best_cost = (
        best_solution.total_cost
        if best_solution.total_cost is not None
        else float("inf")
    )
    reason = ""
    # Iterate with explicit counter to correctly log attempts
    for iters in range(1, params.max_improvement_iterations + 1):
        logger.debug(
            f"\n{Symbols.CHECK} Merge phase iteration {iters}/{params.max_improvement_iterations}"
        )
        selected_clusters = best_solution.selected_clusters
        if selected_clusters.empty:
            logger.info(
                "Initial solution has no selected clusters. Skipping merge phase."
            )
            reason = "initial solution empty"
            break

        # Ensure goods columns exist
        config_lookup = _create_config_lookup(configurations)
        for good in params.goods:
            if good not in selected_clusters.columns:
                selected_clusters[good] = selected_clusters["Config_ID"].map(
                    lambda x: config_lookup[str(x)][good]
                )
        merged_clusters = generate_merge_phase_clusters(
            selected_clusters, configurations, customers_df, params
        )
        if merged_clusters.empty:
            logger.debug("→ No valid merged clusters generated")
            reason = "no candidate merges"
            break

        logger.debug(f"→ Generated {len(merged_clusters)} merged cluster options")
        combined_clusters = pd.concat(
            [selected_clusters, merged_clusters], ignore_index=True
        )
        # Call solver without triggering another merge phase
        internal_params = replace(params, post_optimization=False)

        # Use the internal solver that accepts DataFrames. Force exact optimality
        # (gap = 0) in the improvement iterations to avoid early convergence
        # due to tolerance.
        from fleetmix.optimization.core import _solve_internal

        exact_solver = pick_solver(verbose=True, gap_rel=0.0)
        trial_solution = _solve_internal(
            combined_clusters,
            configurations,
            customers_df,
            internal_params,
            solver=exact_solver,
            verbose=True,
        )
        trial_cost = (
            trial_solution.total_cost
            if trial_solution.total_cost is not None
            else float("inf")
        )
        cost_better = trial_cost < best_cost - 1e-6

        same_choice = False
        if (
            not trial_solution.selected_clusters.empty
            and not best_solution.selected_clusters.empty
        ):
            trial_ids = set(trial_solution.selected_clusters["Cluster_ID"])
            best_ids = set(best_solution.selected_clusters["Cluster_ID"])
            same_choice = trial_ids == best_ids

        logger.debug(
            f"→ Trial cost={trial_cost:.2f}, best cost={best_cost:.2f}, Δ={(trial_cost - best_cost):.2f}"
        )
        if not cost_better:
            reason = "no cost improvement"
            break
        if same_choice:
            reason = "same chosen clusters"
            break

        # Accept improvement and continue
        best_solution = trial_solution
        best_cost = trial_cost
    else:
        # Loop completed without breaks
        reason = "iteration cap reached"
        iters = params.max_improvement_iterations

    logger.debug(f"Merge phase finished after {iters} iteration(s): {reason}")
    return best_solution
