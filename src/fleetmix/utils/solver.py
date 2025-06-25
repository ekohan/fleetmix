"""Solver utilities for FleetMix."""

import importlib.util
import os

import pulp
import pulp.apis

from fleetmix.registry import SOLVER_ADAPTER_REGISTRY, register_solver_adapter


@register_solver_adapter("gurobi")
class GurobiAdapter:
    """Adapter for Gurobi solver."""

    def get_pulp_solver(
        self,
        verbose: bool = False,
        gap_rel: float | None = 0.001,
    ) -> pulp.LpSolver:
        """Return a configured Gurobi solver instance.

        Args:
            verbose: If *True* the solver prints progress messages.
            gap_rel: Relative MIP gap.  ``None`` disables the parameter so the
                solver aims for an exact solution (gap = 0).  CBC accepts the
                same keyword so we keep the signature consistent across
                adapters.
        """
        msg = 1 if verbose else 0
        kwargs: dict[str, float | int] = {"msg": msg}
        # Only pass gapRel when an explicit tolerance is requested – omitting
        # it forces the solver to strive for optimality with gap = 0.
        if gap_rel is not None:
            kwargs["gapRel"] = gap_rel
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
        verbose: bool = False,
        gap_rel: float | None = 0.01,
    ) -> pulp.LpSolver:
        """Return a configured CBC solver instance."""
        msg = 1 if verbose else 0
        kwargs: dict[str, float | int] = {"msg": msg}
        if gap_rel is not None:
            kwargs["gapRel"] = gap_rel
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


def pick_solver(verbose: bool = False, gap_rel: float | None = 0.01):
    """
    Return a PuLP solver instance.
    Priority
    1. FSM_SOLVER env-var: 'gurobi' | 'cbc' | 'auto'
    2. If 'auto' (default): try GUROBI_CMD, fall back to PULP_CBC_CMD.
    """
    choice = os.getenv("FSM_SOLVER", "auto").lower()

    if choice == "gurobi":
        adapter = SOLVER_ADAPTER_REGISTRY["gurobi"]()
        return adapter.get_pulp_solver(verbose=verbose, gap_rel=gap_rel)
    if choice == "cbc":
        adapter = SOLVER_ADAPTER_REGISTRY["cbc"]()
        return adapter.get_pulp_solver(verbose=verbose, gap_rel=gap_rel)

    # auto: try Gurobi, fallback to CBC on instantiation errors
    gurobi_adapter = SOLVER_ADAPTER_REGISTRY["gurobi"]()
    if gurobi_adapter.available:
        try:
            return gurobi_adapter.get_pulp_solver(verbose=verbose, gap_rel=gap_rel)
        except (pulp.PulpError, OSError):
            # Fall back to CBC if Gurobi fails
            pass

    cbc_adapter = SOLVER_ADAPTER_REGISTRY["cbc"]()
    return cbc_adapter.get_pulp_solver(verbose=verbose, gap_rel=gap_rel)
