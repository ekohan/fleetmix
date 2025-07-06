"""Solver utilities for FleetMix."""

import importlib.util
import os
from typing import Any

import pulp
import pulp.apis

from fleetmix.config.params import RuntimeParams
from fleetmix.registry import SOLVER_ADAPTER_REGISTRY, register_solver_adapter


@register_solver_adapter("gurobi")
class GurobiAdapter:
    """Adapter for Gurobi solver."""

    def get_pulp_solver(
        self,
        params: RuntimeParams,
    ) -> pulp.LpSolver:
        """Return a configured Gurobi solver instance.

        Args:
            params: Runtime parameters containing verbose, gap_rel, and time_limit settings.
        """
        msg = 1 if params.verbose else 0
        kwargs: dict[str, Any] = {"msg": msg}
        # Only pass gapRel when an explicit tolerance is requested – omitting
        # it forces the solver to strive for optimality with gap = 0.
        if params.gap_rel is not None:
            kwargs["gapRel"] = params.gap_rel

        options: list[tuple[str, int | float]] = []
        # Use time_limit from params if specified, otherwise default to 3 minutes
        time_limit = params.time_limit if params.time_limit is not None else 180
        if time_limit > 0:  # 0 means no limit
            options.append(("TimeLimit", time_limit))

        if options:
            kwargs["options"] = options

        # Enhanced Gurobi parameters to help escape local optima
        # when dealing with multi-vehicle per customer problems
        # TODO: delete or review this
        """
        if os.getenv("FLEETMIX_ENHANCED_MIP", "1") == "1":
            # Create options list for Gurobi parameters

            # MIPFocus: 1 = focus on finding feasible solutions
            #           2 = focus on proving optimality
            #           3 = focus on improving the best bound
            options.append(("MIPFocus", 1))
            options.append(("Symmetry", 2))

            # Increase solution pool to explore more solutions
            options.append(("PoolSolutions", 10))
            options.append(("PoolSearchMode", 2))  # Find n best solutions

            # More aggressive heuristics
            options.append(("Heuristics", 0.1))  # 10% of time on heuristics

            # Stronger cuts to tighten the formulation
            options.append(("Cuts", 2))  # Aggressive cut generation
            options.append(("TimeLimit", 60))

            # Multiple random seeds for diversity
            seed_str = os.getenv("FLEETMIX_MIP_SEED")
            if seed_str:
                options.append(("Seed", int(seed_str)))

            # Tune for finding good solutions quickly
            options.append(("ImproveStartTime", 10))  # Focus on improving after 10s
            options.append(("ImproveStartGap", 0.1))  # Or when gap < 10%

            #kwargs["options"] = options
        """

        return pulp.GUROBI_CMD(**kwargs)

    @property
    def name(self) -> str:
        """Solver name for logging."""
        return "Gurobi"

    @property
    def available(self) -> bool:
        """Check if Gurobi is available."""
        return importlib.util.find_spec("gurobipy") is not None


@register_solver_adapter("cbc")
class CbcAdapter:
    """Adapter for CBC solver."""

    def get_pulp_solver(
        self,
        params: RuntimeParams,
    ) -> pulp.LpSolver:
        """Return a configured CBC solver instance.

        Args:
            params: Runtime parameters containing verbose, gap_rel, and time_limit settings.
        """
        msg = 1 if params.verbose else 0
        kwargs: dict[str, Any] = {"msg": msg}
        if params.gap_rel is not None:
            kwargs["gapRel"] = params.gap_rel

        # CBC uses maxSeconds for time limit
        if params.time_limit is not None and params.time_limit > 0:
            kwargs["maxSeconds"] = params.time_limit

        return pulp.PULP_CBC_CMD(**kwargs)

    @property
    def name(self) -> str:
        """Solver name for logging."""
        return "CBC"

    @property
    def available(self) -> bool:
        """Check if CBC is available."""
        # CBC is always available as it's bundled with PuLP
        return True


def pick_solver(params: RuntimeParams):
    """
    Return a PuLP solver instance based on RuntimeParams.

    Priority:
    1. FSM_SOLVER env-var: 'gurobi' | 'cbc' | 'auto' (overrides params.solver)
    2. params.solver: 'gurobi' | 'cbc' | 'auto'
    3. If 'auto': try GUROBI_CMD, fall back to PULP_CBC_CMD.
    """
    # Environment variable takes precedence over params
    # TODO: check if env var still relevant
    env_choice = os.getenv("FSM_SOLVER")
    choice = (env_choice or params.solver).lower()

    if choice == "gurobi":
        adapter = SOLVER_ADAPTER_REGISTRY["gurobi"]()
        return adapter.get_pulp_solver(params)
    if choice == "cbc":
        adapter = SOLVER_ADAPTER_REGISTRY["cbc"]()
        return adapter.get_pulp_solver(params)

    # auto: try Gurobi, fallback to CBC on instantiation errors
    gurobi_adapter = SOLVER_ADAPTER_REGISTRY["gurobi"]()
    if gurobi_adapter.available:
        try:
            return gurobi_adapter.get_pulp_solver(params)
        except (pulp.PulpError, OSError):
            # Fall back to CBC if Gurobi fails
            pass

    cbc_adapter = SOLVER_ADAPTER_REGISTRY["cbc"]()
    return cbc_adapter.get_pulp_solver(params)
