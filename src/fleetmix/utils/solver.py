import os, pulp, pulp.apis

def pick_solver(verbose: bool = False):
    """
    Return a PuLP solver instance.
    Priority
    1. FSM_SOLVER env-var: 'gurobi' | 'cbc' | 'auto'
    2. If 'auto' (default): try GUROBI_CMD, fall back to PULP_CBC_CMD.
    """
    choice = os.getenv("FSM_SOLVER", "auto").lower()
    msg = 1 if verbose else 0

    if choice == "gurobi":
        return pulp.GUROBI_CMD(msg=msg)
    if choice == "cbc":
        return pulp.PULP_CBC_CMD(msg=msg)

    # auto: try Gurobi, fallback to CBC on instantiation errors
    try:
        return pulp.GUROBI_CMD(msg=msg)
    except (pulp.PulpError, OSError):
        return pulp.PULP_CBC_CMD(msg=msg) 