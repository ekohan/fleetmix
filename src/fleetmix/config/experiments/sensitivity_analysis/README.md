# Sensitivity Analysis Experiments

**Paper Reference**: Real-World Case Study → Benefits of Using Multi-Compartment Vehicles  
**Datasets**: 70 demand days from `src/fleetmix/benchmarking/datasets/case/`

## Experiment Design

This sensitivity analysis compares **Multi-Compartment Vehicles (MCV)** vs **Single-Compartment Vehicles (SCV)** fleet performance across systematic parameter variations.

Each parameter is tested at **-50%, -20%, 0% (baseline), +20%, +50%** from baseline values.

### Fleet Configurations

**MCV Baseline** (`baselines/baseline_mcv.yaml`):
- 3 vehicle types (A, B, C) that can carry all product types (Dry, Chilled, Frozen)
- Vehicles can configure multiple compartments per route

**SCV Baseline** (`baselines/baseline_scv.yaml`):
- 9 specialized vehicle types (A_Dry, A_Chilled, A_Frozen, B_Dry, B_Chilled, B_Frozen, C_Dry, C_Chilled, C_Frozen)
- Each vehicle type restricted to a single product type

### Baseline Parameter Values

| Parameter              | Value        | Notes                    |
|------------------------|--------------|--------------------------|
| Capacity A             | 700          | Smallest vehicle         |
| Capacity B             | 1300         | Medium vehicle           |
| Capacity C             | 2500         | Largest vehicle          |
| Service Time           | 25 minutes   | Per customer stop        |
| Max Route Duration     | 10 hours     | Maximum shift length     |
| Variable Cost          | 10.0/hour    | Hourly operating cost    |
| Fixed Cost A/B/C       | 80/140/180   | Per vehicle dispatch     |
| Compartment Setup Cost | 10           | Per additional compartment |

## Parameter Variations

### 0. Baselines (`baselines/`)

The baseline configurations representing 0% variation - the reference point for all comparisons.

**2 configs**: 1 MCV baseline + 1 SCV baseline

### 1. Capacity (`capacity/`)

Tests impact of vehicle size on fleet composition and costs.

**8 configs**: 4 MCV variations + 4 SCV variations

Example variations:
- `minus_50`: A=350, B=650, C=1250
- `plus_50`: A=1050, B=1950, C=3750

### 2. Service Time (`service_time/`)

Tests impact of customer service duration on route feasibility and fleet requirements.

**8 configs**: 4 MCV variations + 4 SCV variations

Example variations:
- `minus_50`: 12.5 minutes per customer
- `plus_50`: 37.5 minutes per customer

### 3. Max Route Duration (`max_route_duration/`)

Tests impact of driver shift length constraints on fleet size and utilization.

**8 configs**: 4 MCV variations + 4 SCV variations

Example variations:
- `minus_50`: 5 hours max
- `plus_50`: 15 hours max

### 4. Variable Cost (`variable_cost/`)

Tests impact of hourly operating costs on fleet economic trade-offs.

**8 configs**: 4 MCV variations + 4 SCV variations

Example variations:
- `minus_50`: 5.0/hour
- `plus_50`: 15.0/hour

### 5. Feature Impact (`feature_impact/`)

Tests impact of specific algorithm features on solution quality.

**3 configs**:
- `mcv_no_post_optimization.yaml` - Disables iterative improvement phase
- `scv_no_post_optimization.yaml` - Disables iterative improvement phase
- `mcv_no_split_stops.yaml` - Prevents customers from being served by multiple vehicles

## Results in Paper

The sensitivity analysis results using these configs are presented in the paper:

**Key metrics reported:**
- Fleet size (number of vehicles)
- Total operational cost (fixed + variable)
- Load vehicle utilization (%)
- Customers served per vehicle
- Effective route duration (hours)

**Parameters analyzed in detail:** `capacity/`, `service_time/`, `max_route_duration/`

## Running Experiments

### Single Configuration
```bash
fleetmix benchmark case \
    --config src/fleetmix/config/experiments/sensitivity_analysis/capacity/mcv_capacity_plus_20.yaml \
    --format json \
    --output results/capacity_plus_20/
```

### Batch Run - All Capacity Variations
```bash
for config in src/fleetmix/config/experiments/sensitivity_analysis/capacity/*.yaml; do
    config_name=$(basename "$config" .yaml)
    fleetmix benchmark case \
        --config "$config" \
        --format json \
        --output "results/capacity/$config_name/"
done
```

### Batch Run - Only MCV Configs
```bash
for config in src/fleetmix/config/experiments/sensitivity_analysis/capacity/mcv_*.yaml; do
    config_name=$(basename "$config" .yaml)
    fleetmix benchmark case \
        --config "$config" \
        --format json \
        --output "results/capacity/$config_name/"
done
```

### Run All Sensitivity Analysis Experiments
```bash
for param_dir in capacity service_time max_route_duration variable_cost; do
    for config in src/fleetmix/config/experiments/sensitivity_analysis/$param_dir/*.yaml; do
        config_name=$(basename "$config" .yaml)
        fleetmix benchmark case \
            --config "$config" \
            --format json \
            --output "results/$param_dir/$config_name/"
    done
done
```

## Configuration Details

All sensitivity analysis configs share:
- **Demand file**: `sales_2024_avg_day_demand.csv`
- **Clustering method**: `combine` (hybrid approach)
- **Route time estimation**: `BHH` (Beardwood-Halton-Hammersley)
- **Solver**: Gurobi with 0.5% relative gap tolerance
- **Time limit**: 600 seconds (10 minutes)
- **Post-optimization**: Enabled (except in feature_impact tests)
- **Split stops**: Enabled (except in specific feature_impact test)

## File Count Summary

- **Total configs**: 37
  - Baselines: 2 (1 MCV + 1 SCV) - 0% variation
  - Capacity: 8 (4 MCV + 4 SCV) - ±20%, ±50% variations
  - Service time: 8 (4 MCV + 4 SCV) - ±20%, ±50% variations
  - Max route duration: 8 (4 MCV + 4 SCV) - ±20%, ±50% variations
  - Variable cost: 8 (4 MCV + 4 SCV) - ±20%, ±50% variations
  - Feature impact: 3 (2 MCV + 1 SCV) - feature toggle tests

