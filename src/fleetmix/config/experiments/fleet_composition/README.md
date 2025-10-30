# Fleet Composition Analysis

**Paper Reference**: Real-World Case Study → Fleet Composition Analysis

## Overview

Base configuration for fleet composition experiments comparing MCV-only, SCV-only, and mixed fleet performance across varying cost structures.

## Experimental Design

This config serves as the baseline template for a 2-factor experiment (α × C grid):
- **α (Vehicle Premium)**: MCV fixed cost multiplier ∈ [1.0, 2.0] in 11 steps
- **C (Compartment Setup Cost)**: Additional cost per compartment ∈ [0%, 50%] in 6 steps
- **Blocks**: 70 historical demand days (June–August 2024)
- **Total observations**: 67 treatments × 70 days = 4,690 per experiment

See `docs/experimental_design.md` for full methodological details.

## Configuration

**`base_config.yaml`** - Template configuration used by experiment scripts

The scripts in `src/fleetmix/experiments/alpha_analysis/` programmatically generate:
- SCV-only fleets (one specialized vehicle per product type)
- MCV homogeneous fleets (with varying α and C)
- Mixed fleets (SCV + MCV with endogenous selection)

## Usage

This config is not meant to be run directly. It's loaded programmatically by:
- `experiments/alpha_analysis/fleet_templates.py`
- `experiments/alpha_analysis/run_grid.py`
- `experiments/alpha_analysis/run_grid_mixed.py`
- `experiments/alpha_analysis/run_day.py`

