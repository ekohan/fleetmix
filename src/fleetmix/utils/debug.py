"""MILP model debugging utilities for FleetMix."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pulp

from fleetmix.utils.logging import FleetmixLogger

logger = FleetmixLogger.get_logger(__name__)


class ModelDebugger:
    """Solver-agnostic MILP model debugger.

    Provides a clean interface for dumping model artifacts (LP files, MPS files,
    solver logs, IIS) without coupling the optimization code to specific solvers.

    Example:
        # Enable debugging once at startup
        ModelDebugger.enable(debug_dir="debug_output")

        # In optimization code, just call dump after creating/solving
        model = create_optimization_model(...)
        model.solve(solver)
        ModelDebugger.dump(model, "fsm_model")
    """

    active: bool = False
    _dir: Path
    _artifacts: set[str]

    @classmethod
    def enable(
        cls, debug_dir: Path | str = ".", artifacts: set[str] | None = None
    ) -> None:
        """Enable MILP debugging and configure output directory.

        Args:
            debug_dir: Directory to write debug artifacts to
            artifacts: Set of artifacts to generate. Defaults to {"lp", "mps", "solver_log", "iis"}
        """
        cls.active = True
        cls._dir = Path(debug_dir).expanduser().resolve()
        cls._dir.mkdir(parents=True, exist_ok=True)
        cls._artifacts = artifacts or {"lp", "mps", "solver_log", "iis"}
        logger.info(f"⚙ MILP debugging enabled → {cls._dir}")
        logger.info(f"  Artifacts: {', '.join(sorted(cls._artifacts))}")

    @classmethod
    def dump(cls, model: pulp.LpProblem, name: str = "model") -> None:
        """Dump model artifacts if debugging is enabled.

        Args:
            model: The PuLP model to dump
            name: Base name for output files (without extension)
        """
        if not cls.active:
            return

        base = cls._dir / name

        # Standard LP format (always available)
        if "lp" in cls._artifacts:
            try:
                model.writeLP(str(base.with_suffix(".lp")))
                logger.debug(f"Wrote LP file: {base}.lp")
            except Exception as e:
                logger.warning(f"Failed to write LP file: {e}")

        # MPS format (may not be available for all model types)
        if "mps" in cls._artifacts:
            try:
                model.writeMPS(str(base.with_suffix(".mps")))
                logger.debug(f"Wrote MPS file: {base}.mps")
            except Exception:
                # MPS writing can fail for various reasons, silently skip
                pass

        # Capture solver log by re-solving with verbose output
        if "solver_log" in cls._artifacts and model.solver is not None:
            cls._capture_solver_log(model, str(base.with_suffix(".log")))

        # Extract IIS if solver supports it (e.g., Gurobi)
        if "iis" in cls._artifacts and model.status in [
            pulp.LpStatusInfeasible,
            pulp.LpStatusNotSolved,
        ]:
            cls._extract_iis(model, str(base.with_suffix(".iis")))

    @staticmethod
    def _capture_solver_log(model: pulp.LpProblem, path: str) -> None:
        """Capture solver output by re-solving with verbose mode."""
        solver = model.solver
        if solver is None:
            return

        # Save original verbosity setting
        old_msg = getattr(solver, "msg", 1)

        try:
            # Force verbose output
            solver.msg = 1

            # Capture stdout during solve
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                model.solve(solver)

            # Write captured output
            Path(path).write_text(buf.getvalue())
            logger.debug(f"Captured solver log: {path}")

        except Exception as e:
            logger.warning(f"Failed to capture solver log: {e}")
        finally:
            # Restore original verbosity
            solver.msg = old_msg

    @staticmethod
    def _extract_iis(model: pulp.LpProblem, path: str) -> None:
        """Try to extract an Irreducible Infeasible Set (IIS).

        This only works for solvers that expose a .solverModel attribute
        with IIS computation capabilities (e.g., Gurobi).
        """
        # Check if solver has the required interface
        solver_model = getattr(model.solver, "solverModel", None)
        if solver_model is None:
            return

        try:
            # Attempt to compute IIS (Gurobi-specific but wrapped safely)
            if hasattr(solver_model, "computeIIS"):
                solver_model.computeIIS()
                solver_model.write(path)
                logger.info(f"Extracted IIS: {path}")
        except Exception as e:
            # IIS extraction is best-effort, don't fail if unsupported
            logger.debug(f"IIS extraction not available: {e}")
