"""Solver utilities for FleetMix."""
import os
import pulp
import pulp.apis

from fleetmix.registry import register_solver_adapter, SOLVER_ADAPTER_REGISTRY


@register_solver_adapter('gurobi')
class GurobiAdapter:
    """Adapter for Gurobi solver."""
    
    def get_pulp_solver(self, verbose: bool = False) -> pulp.LpSolver:
        """Return configured Gurobi solver instance."""
        msg = 1 if verbose else 0
        return pulp.GUROBI_CMD(msg=msg)
    
    @property
    def name(self) -> str:
        """Solver name for logging."""
        return "Gurobi"
    
    @property
    def available(self) -> bool:
        """Check if Gurobi is available."""
        try:
            # Check availability without creating a full solver instance
            import gurobipy
            return True
        except ImportError:
            return False


@register_solver_adapter('cbc')
class CbcAdapter:
    """Adapter for CBC solver."""
    
    def get_pulp_solver(self, verbose: bool = False) -> pulp.LpSolver:
        """Return configured CBC solver instance."""
        msg = 1 if verbose else 0
        return pulp.PULP_CBC_CMD(msg=msg)
    
    @property
    def name(self) -> str:
        """Solver name for logging."""
        return "CBC"
    
    @property
    def available(self) -> bool:
        """Check if CBC is available."""
        # CBC is always available as it's bundled with PuLP
        return True


def pick_solver(verbose: bool = False):
    """
    Return a PuLP solver instance.
    Priority
    1. FSM_SOLVER env-var: 'gurobi' | 'cbc' | 'auto'
    2. If 'auto' (default): try GUROBI_CMD, fall back to PULP_CBC_CMD.
    """
    choice = os.getenv("FSM_SOLVER", "auto").lower()

    if choice == "gurobi":
        adapter = SOLVER_ADAPTER_REGISTRY['gurobi']()
        return adapter.get_pulp_solver(verbose)
    if choice == "cbc":
        adapter = SOLVER_ADAPTER_REGISTRY['cbc']()
        return adapter.get_pulp_solver(verbose)

    # auto: try Gurobi, fallback to CBC on instantiation errors
    gurobi_adapter = SOLVER_ADAPTER_REGISTRY['gurobi']()
    if gurobi_adapter.available:
        try:
            return gurobi_adapter.get_pulp_solver(verbose)
        except (pulp.PulpError, OSError):
            # Fall back to CBC if Gurobi fails
            pass
    
    cbc_adapter = SOLVER_ADAPTER_REGISTRY['cbc']()
    return cbc_adapter.get_pulp_solver(verbose) 