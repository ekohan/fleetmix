# Reproducing Paper Experiments

This guide explains how to reproduce all experiments from the FleetMix paper using the `fleetmix reproduce-paper` command.

## Overview

The FleetMix paper presents three main experimental sections:

1. **MCVRP Benchmark Instances** - Validates the matheuristic approach on synthetic instances (Section §6. Effectiveness of the Matheuristic Approach)
2. **Sensitivity Analysis** - Analyzes the impact of operational parameters on fleet performance (Section §7.2. Benefits of Using Multi-Compartment Vehicles)
3. **Fleet Composition Analysis** - Studies how cost structure affects fleet composition decisions (Section §7.3. Impact of Cost Structure on Fleet Composition)

Each experiment can be reproduced using dedicated CLI commands with full control over execution parameters.

## Quick Start

```bash
# View all available reproduce-paper commands
fleetmix reproduce-paper --help

# List available MCVRP instances
fleetmix reproduce-paper mcvrp-instances --list

# Run all experiments (warning: this will take many hours)
fleetmix reproduce-paper mcvrp-instances
fleetmix reproduce-paper sensitivity-analysis
fleetmix reproduce-paper fleet-composition
```

## Experiment 1: MCVRP Benchmark Instances

**Paper Section:** 6. Effectiveness of the Matheuristic Approach (Table 2)

**Purpose:** Validate the matheuristic by comparing against published results from Henke (2015, 2019).

### Available Instances

- 150 instances from Henke (2015): 10 customers with supply parameters s ∈ {1,2,3}
- 3 larger instances from Henke (2015): 50 customers
- 48 instances from Henke (2019): 10-50 customers with s=3

Total: ~198 instances

### Commands

```bash
# List all available instances
fleetmix reproduce-paper mcvrp-instances --list

# Run all instances (takes ~20 minutes)
fleetmix reproduce-paper mcvrp-instances

# Run specific instances
fleetmix reproduce-paper mcvrp-instances --instances "2015_10_3_3_1_(01),2015_10_3_3_1_(02)"

# Run and skip existing results
fleetmix reproduce-paper mcvrp-instances --skip-existing
```

### Configuration

- Default config: `src/fleetmix/config/experiments/synthetic_test_instances/base_config.yaml`
- Homogeneous fleet (single vehicle type)
- No route duration constraints (to match Henke benchmark)
- Multi-stop delivery policy enabled

### Output

Results are saved as JSON files with statistics on the instance run:

```
results/paper/mcvrp_instances/
  mcvrp_2015_10_3_3_1_(01).json
  mcvrp_2015_10_3_3_1_(02).json
  ...
```

## Experiment 2: Sensitivity Analysis

**Paper Section:** 7.2. Benefits of Using Multi-Compartment Vehicles (Figure 3, Table 3)

**Purpose:** Compare MCV vs SCV fleet performance across systematic parameter variations.

### Parameters Tested

Each parameter is varied at ±50%, ±20%, and baseline (0%):

- **Capacity**: Vehicle load capacity
- **Service Time**: Customer service duration
- **Max Route Duration**: Driver shift length
- **Variable Cost**: Hourly operating cost (optional)

### Fleet Types

- **MCV**: 3 vehicle types (A, B, C) with flexible compartments
- **SCV**: 9 specialized vehicle types (one per product type)

### Commands

```bash
# Run all sensitivity experiments (2,590 runs, ~8 hours with 8 cores)
fleetmix reproduce-paper sensitivity-analysis

# Run only capacity variations
fleetmix reproduce-paper sensitivity-analysis --parameters capacity

# Run only MCV fleet
fleetmix reproduce-paper sensitivity-analysis --fleet-types mcv

# Run baseline only (for testing)
fleetmix reproduce-paper sensitivity-analysis --variations baseline

# Run specific demand days
fleetmix reproduce-paper sensitivity-analysis --demand-days "sales_2024-07-02_demand,sales_2024-07-03_demand"

# Run and skip existing results
fleetmix reproduce-paper sensitivity-analysis --skip-existing
```

### Configuration

Base configurations are in:
- `src/fleetmix/config/experiments/sensitivity_analysis/baselines/`
- `src/fleetmix/config/experiments/sensitivity_analysis/{parameter}/`

Each variation has separate YAML configs for MCV and SCV fleets.

### Demand Data

70 real demand days from: `src/fleetmix/benchmarking/datasets/case/sales_*.csv`

### Output

```
results/paper/sensitivity_analysis/
  capacity/
    mcv_capacity_minus_50/
      sales_2024-06-01_demand.json
      ...
    scv_capacity_minus_50/
      ...
  service_time/
    ...
  max_route_duration/
    ...
  summary.parquet  # Aggregated results
```

### Key Metrics

The summary includes:
- Total cost (fixed + variable)
- Fleet size
- Vehicle utilization
- Customers per vehicle
- Route duration
- Fleet composition by vehicle type

## Experiment 3: Fleet Composition Analysis

**Paper Section:** 7.3. Impact of Cost Structure on Fleet Composition (Figure 4, Table 4)

**Purpose:** Analyze how MCV adoption varies across different cost structures.

### Parameter Grid

- **Alpha (α)**: MCV fixed cost multiplier [1.0 to 2.0, 11 values]
- **C**: Compartment setup cost [0 to 50, 6 values]
- **Demand days**: 70 real demand instances

Total runs: 11 × 6 × 70 = 4,620 mixed fleet + 70 SCV baselines = 4,690 runs

### Commands

```bash
# Run full grid (4,690 runs, ~48 hours with 16 cores)
fleetmix reproduce-paper fleet-composition

# Run with custom grid (smaller)
fleetmix reproduce-paper fleet-composition \
  --alpha-grid "1.0,1.2,1.4,1.6" \
  --c-values "0,10,20"

# Run specific demand days
fleetmix reproduce-paper fleet-composition \
  --demand-days "sales_2024-07-02_demand,sales_2024-07-03_demand"

```

### Configuration

Base config: `src/fleetmix/config/experiments/fleet_composition/base_config.yaml`

Fleet templates are dynamically generated using:
- `src/fleetmix/experiments/alpha_analysis/fleet_templates.py`

### Output

```
results/paper/fleet_composition/
  raw/
    sales_2024-06-01_demand_SCV_BASE.json
    sales_2024-06-01_demand_MIXED_1.00_0.json
    sales_2024-06-01_demand_MIXED_1.10_0.json
    ...
  summary_mixed.parquet  # Aggregated results with deltas
```

## Expected Runtimes

Approximate runtimes on a standard workstation:

| Experiment | Sequential |
|------------|-----------|
| MCVRP instances (198) | ~30 min |
| Sensitivity analysis (1820) | ~30 hours |
| Fleet composition (4,690) | ~80 hours |

