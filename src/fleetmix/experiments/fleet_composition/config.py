"""
Experiment configuration for alpha analysis.

This module defines the grids for alpha and C, and paths to demand files.
"""

from pathlib import Path

import numpy as np

__all__ = ["ALPHA_GRID", "C_VALUES", "DEMAND_DIR", "DEMAND_FILES", "HANDOVER_SPEC"]

# Alpha grid: fixed-cost multiplier for MCV vs SCV
ALPHA_GRID = np.linspace(1.0, 2.0, 11).tolist()

# C grid: compartment setup cost
C_VALUES = np.linspace(0, 50, 6).tolist()

# Demand files path constants
DEMAND_DIR = Path("src/fleetmix/benchmarking/datasets/case").resolve()

# Specific demand files (using synthetic data)
DEMAND_FILES = sorted(list(DEMAND_DIR.glob("synthetic_sales_*.csv")))


# Handover spec (can be exported to JSON)
HANDOVER_SPEC = {"alpha": ALPHA_GRID, "C_values": C_VALUES}
