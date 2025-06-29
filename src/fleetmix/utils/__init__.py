"""
Utility helpers for Fleetmix (dependency-free).

• Route-time estimation (`route_time.py`) – wraps BHH, legacy heuristics and a PyVRP TSP fallback.
• Command-line interface helpers (`cli.py`).
• File I/O (`save_results.py`, `data_processing.py`).
• Logging colour codes and progress bars (`logging.py`).
• Solver adapter (`solver.py`) – picks CBC / Gurobi / CPLEX based on the runtime environment.
• Generation of vehicle configurations (`vehicle_configurations.py`).
• MILP model debugging (`debug.py`) – solver-agnostic model artifact dumping.

"""

from .debug import ModelDebugger
from .project_root import PROJECT_ROOT, get_project_root

__all__ = ["PROJECT_ROOT", "get_project_root", "ModelDebugger"]
