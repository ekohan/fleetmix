"""Solver utilities for FleetMix."""

import importlib.util
import os
from typing import Any

import pulp
import pulp.apis

from fleetmix.config.params import RuntimeParams
from fleetmix.registry import SOLVER_ADAPTER_REGISTRY, register_solver_adapter


def extract_optimality_gap(model, solver) -> float | None:
    """
    Try to fetch the *relative* optimality gap from a PuLP model and solver instance.

    Tries multiple approaches:
    1. Model attributes (solutionGap)
    2. Solver-specific extraction (primarily GUROBI_CMD log parsing)

    The value is returned in **percentage points** (e.g. ``0.48`` for a 0.48 % gap).
    For solvers that cannot report a gap the function returns ``None``.

    Args:
        model: The PuLP model instance after solving
        solver: The PuLP solver instance used

    Returns:
        Optimality gap in percentage points, or None if unavailable
    """
    # TODO: toggle as runtime/io feature
    import pulp

    # ------------------------------------------------------------------
    # Try to get gap from model attributes first
    # ------------------------------------------------------------------
    if hasattr(model, "solutionGap"):
        optimality_gap = model.solutionGap
        if optimality_gap is not None and optimality_gap <= 1.0:
            return float(optimality_gap * 100.0)  # Convert to percentage
        elif optimality_gap is not None:
            return float(optimality_gap)  # Already in percentage

    # ------------------------------------------------------------------
    # Gurobi via PuLP log parsing
    # ------------------------------------------------------------------
    if hasattr(pulp, "GUROBI_CMD") and isinstance(solver, pulp.GUROBI_CMD):
        # Parse gap from Gurobi log file
        import os
        import re

        # Try multiple log locations
        log_paths = []

        # Current working directory (when keepFiles=True)
        log_paths.append("gurobi.log")

        # Solver's temporary directory
        if hasattr(solver, "tmpDir") and solver.tmpDir:
            log_paths.append(os.path.join(solver.tmpDir, "gurobi.log"))

        # Search for gap in each potential log file
        for log_path in log_paths:
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        content = f.read()

                    # Look for the final gap report in the log
                    # Pattern: "Best objective 7.900174045037e+04, best bound 7.850209502181e+04, gap 0.6324%"
                    gap_pattern = (
                        r"Best objective [^,]+, best bound [^,]+, gap (\d+\.?\d*)%"
                    )
                    matches = re.findall(gap_pattern, content, re.IGNORECASE)

                    if matches:
                        # Take the last match (final gap at termination)
                        gap_str = matches[-1]
                        return float(gap_str)

                except Exception:  # noqa: BLE001
                    # If file reading fails, try next location
                    continue

    # ------------------------------------------------------------------
    # Other solvers – nothing to report
    # ------------------------------------------------------------------
    return None


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
        kwargs: dict[str, Any] = {"msg": msg, "keepFiles": True}
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

        if params.time_limit is not None and params.time_limit > 0:
            kwargs["timeLimit"] = params.time_limit

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
