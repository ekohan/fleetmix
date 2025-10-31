# FleetMix Experiment Configurations

This directory contains all configuration files used for experiments reported in the research paper.

## Structure

```
experiments/
├── sensitivity_analysis/       # Parameter sensitivity analysis
│   ├── baselines/              # Baseline configurations (0% variation)
│   ├── capacity/               # Vehicle capacity variations
│   ├── service_time/           # Customer service time variations
│   ├── max_route_duration/     # Maximum route duration variations
│   ├── variable_cost/          # Variable operating cost variations
│   └── feature_impact/         # Tests with algorithm features disabled
│
└── fleet_composition/          # Fleet composition analysis (α × C grid)
    └── base_config.yaml        # Template for MCV/SCV/Mixed fleet experiments
```

## Experiments

### Sensitivity Analysis

Parameter sensitivity testing at **-50%, -20%, 0% (baseline), +20%, +50%** variations, comparing MCV (Multi-Compartment Vehicles) vs SCV (Single-Compartment Vehicles) fleet performance across 70 demand instances.

See `sensitivity_analysis/README.md` for detailed documentation.

### Fleet Composition Analysis

Two-factor experiment (α × C grid) comparing MCV-only, SCV-only, and mixed fleet performance across varying cost structures. Base configuration is used programmatically by experiment scripts to generate fleet configurations.

See `fleet_composition/README.md` for details and `docs/experimental_design.md` for methodology.

## Usage

```bash
# Run baseline configuration
fleetmix benchmark case --config src/fleetmix/config/experiments/sensitivity_analysis/baselines/baseline_mcv.yaml

# Run all configs in a category
for config in src/fleetmix/config/experiments/sensitivity_analysis/capacity/*.yaml; do
    fleetmix benchmark case --config "$config" --format json
done
```

## Paper Reference

These experiments are reported in the paper under:
- Real-World Case Study → Benefits of Using Multi-Compartment Vehicles
- Real-World Case Study → Fleet Composition Analysis

