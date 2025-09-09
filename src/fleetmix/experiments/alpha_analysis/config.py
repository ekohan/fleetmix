"""
Experiment configuration for alpha analysis.

This module defines the grids for alpha and C, and paths to demand files.
"""

from pathlib import Path

import numpy as np

# Alpha grid: fixed-cost multiplier for MCV vs SCV
ALPHA_GRID = np.linspace(1.0, 2.0, 11).tolist()

# C grid: compartment setup cost
C_VALUES = np.linspace(0, 50, 6).tolist()

# Demand files path constants
DEMAND_DIR = Path("src/fleetmix/benchmarking/datasets/case").resolve()

# Specific demand files as provided
DEMAND_FILES = [
    DEMAND_DIR / "sales_2024-06-01_demand.csv",
    DEMAND_DIR / "sales_2024-06-04_demand.csv",
    DEMAND_DIR / "sales_2024-06-05_demand.csv",
    DEMAND_DIR / "sales_2024-06-06_demand.csv",
    DEMAND_DIR / "sales_2024-06-07_demand.csv",
    DEMAND_DIR / "sales_2024-06-08_demand.csv",
    DEMAND_DIR / "sales_2024-06-11_demand.csv",
    DEMAND_DIR / "sales_2024-06-12_demand.csv",
    DEMAND_DIR / "sales_2024-06-13_demand.csv",
    DEMAND_DIR / "sales_2024-06-14_demand.csv",
    DEMAND_DIR / "sales_2024-06-15_demand.csv",
    DEMAND_DIR / "sales_2024-06-17_demand.csv",
    DEMAND_DIR / "sales_2024-06-18_demand.csv",
    DEMAND_DIR / "sales_2024-06-19_demand.csv",
    DEMAND_DIR / "sales_2024-06-20_demand.csv",
    DEMAND_DIR / "sales_2024-06-21_demand.csv",
    DEMAND_DIR / "sales_2024-06-22_demand.csv",
    DEMAND_DIR / "sales_2024-06-24_demand.csv",
    DEMAND_DIR / "sales_2024-06-25_demand.csv",
    DEMAND_DIR / "sales_2024-06-26_demand.csv",
    DEMAND_DIR / "sales_2024-06-27_demand.csv",
    DEMAND_DIR / "sales_2024-06-29_demand.csv",
    DEMAND_DIR / "sales_2024-07-02_demand.csv",
    DEMAND_DIR / "sales_2024-07-03_demand.csv",
    DEMAND_DIR / "sales_2024-07-04_demand.csv",
    DEMAND_DIR / "sales_2024-07-05_demand.csv",
    DEMAND_DIR / "sales_2024-07-06_demand.csv",
    DEMAND_DIR / "sales_2024-07-08_demand.csv",
    DEMAND_DIR / "sales_2024-07-09_demand.csv",
    DEMAND_DIR / "sales_2024-07-10_demand.csv",
    DEMAND_DIR / "sales_2024-07-11_demand.csv",
    DEMAND_DIR / "sales_2024-07-12_demand.csv",
    DEMAND_DIR / "sales_2024-07-13_demand.csv",
    DEMAND_DIR / "sales_2024-07-16_demand.csv",
    DEMAND_DIR / "sales_2024-07-17_demand.csv",
    DEMAND_DIR / "sales_2024-07-18_demand.csv",
    DEMAND_DIR / "sales_2024-07-19_demand.csv",
    DEMAND_DIR / "sales_2024-07-22_demand.csv",
    DEMAND_DIR / "sales_2024-07-23_demand.csv",
    DEMAND_DIR / "sales_2024-07-24_demand.csv",
    DEMAND_DIR / "sales_2024-07-25_demand.csv",
    DEMAND_DIR / "sales_2024-07-26_demand.csv",
    DEMAND_DIR / "sales_2024-07-27_demand.csv",
    DEMAND_DIR / "sales_2024-07-29_demand.csv",
    DEMAND_DIR / "sales_2024-07-30_demand.csv",
    DEMAND_DIR / "sales_2024-07-31_demand.csv",
    DEMAND_DIR / "sales_2024-08-01_demand.csv",
    DEMAND_DIR / "sales_2024-08-02_demand.csv",
    DEMAND_DIR / "sales_2024-08-03_demand.csv",
    DEMAND_DIR / "sales_2024-08-05_demand.csv",
    DEMAND_DIR / "sales_2024-08-07_demand.csv",
    DEMAND_DIR / "sales_2024-08-08_demand.csv",
    DEMAND_DIR / "sales_2024-08-09_demand.csv",
    DEMAND_DIR / "sales_2024-08-10_demand.csv",
    DEMAND_DIR / "sales_2024-08-12_demand.csv",
    DEMAND_DIR / "sales_2024-08-13_demand.csv",
    DEMAND_DIR / "sales_2024-08-14_demand.csv",
    DEMAND_DIR / "sales_2024-08-15_demand.csv",
    DEMAND_DIR / "sales_2024-08-16_demand.csv",
    DEMAND_DIR / "sales_2024-08-17_demand.csv",
    DEMAND_DIR / "sales_2024-08-20_demand.csv",
    DEMAND_DIR / "sales_2024-08-21_demand.csv",
    DEMAND_DIR / "sales_2024-08-22_demand.csv",
    DEMAND_DIR / "sales_2024-08-23_demand.csv",
    DEMAND_DIR / "sales_2024-08-24_demand.csv",
    DEMAND_DIR / "sales_2024-08-26_demand.csv",
    DEMAND_DIR / "sales_2024-08-27_demand.csv",
    DEMAND_DIR / "sales_2024-08-28_demand.csv",
    DEMAND_DIR / "sales_2024-08-29_demand.csv",
    DEMAND_DIR / "sales_2024-08-31_demand.csv",
]

# Handover spec (can be exported to JSON)
HANDOVER_SPEC = {"alpha": ALPHA_GRID, "C_values": C_VALUES}
