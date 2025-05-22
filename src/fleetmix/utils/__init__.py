"""
Utility helpers for MIT Fleet Design Optimisation.

This sub-package groups together reusable components that are *orthogonal* to the core
optimisation logic but essential for a production-grade workflow:

• Route-time estimation (`route_time.py`) – wraps BHH, legacy heuristics and a PyVRP TSP fallback.
• Command-line interface helpers (`cli.py`).
• File I/O (`save_results.py`, `data_processing.py`).
• Logging colour codes and progress bars (`logging.py`).
• Solver adapter (`solver.py`) – picks CBC / Gurobi / CPLEX based on the runtime environment.
• Generation of vehicle configurations (`vehicle_configurations.py`).

These modules are **dependency-free** beyond the scientific Python stack so they can be imported in
isolation by other projects.
"""

from .project_root import PROJECT_ROOT, get_project_root

__all__ = [
    "PROJECT_ROOT", 
    "get_project_root"
]
