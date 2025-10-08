# FleetMix User Guide

> **Complete guide for practitioners solving real-world fleet design problems**

---

## Getting Started

### Installation

```bash
uv pip install fleetmix
```

Or from source:
```bash
git clone https://github.com/ekohan/fleetmix.git
cd fleetmix
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv fleetmix-env
source fleetmix-env/bin/activate
uv sync --all-extras
```

### Your First Fleet Size & Mix (FSM) Optimization

**TODO**: be more clear here on data layout. after Pydantic improvements.
**Step 1**: Prepare your customer demand data (CSV format)

```csv
customer_id,Latitude,Longitude,demand_Dry,demand_Chilled,demand_Frozen
C001,4.6097,-74.0817,150,80,40
C002,4.6512,-74.1120,200,100,50
C003,4.7234,-74.0654,100,60,30
...
```

**Step 2**: Create a configuration file

```yaml
# my_fleet_config.yaml
vehicles:
  SmallTruck:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
  
  LargeTruck:
    capacity: 4500
    fixed_cost: 225
    avg_speed: 30
    service_time: 25
    max_route_time: 12

goods:
  - Dry
  - Chilled
  - Frozen

depot:
  latitude: 4.6097
  longitude: -74.0817
```

**Step 3**: Run optimization

```bash
fleetmix optimize \
  --demand customers.csv \
  --config my_fleet_config.yaml \
  --output results/ \
  --format excel
```

**Step 4**: Review results

Results saved to `results/fleet_solution_TIMESTAMP.xlsx` with sheets:
- **Summary**: Total cost, vehicle count, utilization
- **Fleet Composition**: Vehicles selected, configuration details
- **Routes**: Customer assignments, route times
- **Costs**: Breakdown of fixed vs variable costs

---

## Understanding Your Results

### Key Metrics

**Total Cost**: Sum of fixed (daily vehicle) and variable (distance/time) costs

**Fleet Composition**: Number and type of vehicles recommended

**Utilization**: How full vehicles are on average

**Route Times**: Average and maximum route duration
- Compare against `max_route_time` to see constraint tightness

### What is a "Vehicle Configuration"?

A vehicle configuration = vehicle type + compartment setup

Example configurations for one truck:
1. Truck with only Dry compartment (single-compartment)
2. Truck with Dry + Chilled compartments (multi-compartment)
3. Truck with all three compartments (full multi-compartment)

**FleetMix automatically selects** which configurations to use based on costs and demand.

---

## Common Scenarios

**TODO** check scenario code is valid.
### Scenario 1: Cost Reduction Analysis

**Question**: "Should I invest in multi-compartment vehicles?"

**Approach**:
1. Run with current fleet (single-compartment):
```yaml
vehicles:
  DryTruck:
    allowed_goods: ["Dry"]
  ChilledTruck:
    allowed_goods: ["Chilled"]
  FrozenTruck:
    allowed_goods: ["Frozen"]
```

2. Run with multi-compartment option:
```yaml
vehicles:
  MultiTempTruck:
    # No allowed_goods = can carry all
```

3. Compare costs and fleet sizes

**Typical finding**: 15-40% cost reduction with MCVs (Paper §6)

### Scenario 2: Fleet Size Planning

**Question**: "How many vehicles do I need for varying demand levels?"

**Approach**: Parameter sweep over demand
```python
import fleetmix as fm
import pandas as pd

results = []
for demand_multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
    # Scale demand
    customers = load_customers()
    customers[['demand_Dry', 'demand_Chilled', 'demand_Frozen']] *= demand_multiplier
    
    # Optimize
    solution = fm.optimize(demand=customers, config="config.yaml")
    
    results.append({
        'demand_level': demand_multiplier,
        'vehicles_needed': sum(solution.vehicles_used.values()),
        'total_cost': solution.total_cost,
    })

# Plot results
df = pd.DataFrame(results)
df.plot(x='demand_level', y='vehicles_needed')
```

### Scenario 3: Service Level Analysis

**Question**: "What if I reduce max route time from 10h to 8h?"

**Approach**: Vary `max_route_time` parameter
```python
for max_time in [6, 7, 8, 9, 10, 11, 12]:
    config = load_config()
    for vehicle in config['vehicles'].values():
        vehicle['max_route_time'] = max_time
    
    solution = fm.optimize(demand=customers, config=config)
    print(f"{max_time}h: {sum(solution.vehicles_used.values())} vehicles")
```

**Expected**: Tighter time limits require more vehicles

---

## Tuning Performance

### For Faster Results (Large Problems)

1. **Use fewer clustering methods**:
```yaml
clustering:
  methods:
    - minibatch_kmeans  # Fastest
```

2. **Reduce cluster variants**:
```yaml
clustering:
  geo_weight: [1.0]  # Just one value
  n_clusters_range: [auto]  # Skip manual values
```

3. **Use BHH route time (default)**:
```yaml
route_time:
  method: bhh  # Much faster than 'tsp'
```

4. **Set solver time limit**:
```yaml
optimization:
  time_limit: 60  # Acceptable for approximate solution
  mip_gap: 0.005  # 0.5% gap OK for large problems
```

5. **Tune merging aggressiveness**:
```yaml
# Conservative (faster): merge only very small clusters
small_cluster_size: 3
nearest_merge_candidates: 5

# Aggressive (slower, better quality): merge more clusters  
small_cluster_size: 15
nearest_merge_candidates: 20
```

---

## Troubleshooting

### "No feasible solution found"

**Causes**:
- Customer demand exceeds largest vehicle capacity
- Route time constraints too tight (max_route_time too low)
- Geographic spread too large for vehicle speeds

**Solutions**:
- Increase vehicle capacities
- Increase `max_route_time`
- Add more vehicle types
- Check demand units (kg vs tons?)

**Checklist**:
- Units consistency (all in kg? all in km/h?)
- Cost parameters (fixed vs variable balance)
- Geographic coordinates (latitude/longitude correct?)
- Demand values (not accidentally scaled?)

## Getting Help

- **Documentation**: Start with [quickstart.md](quickstart.md)
- **Technical Details**: See [ARCHITECTURE.md](ARCHITECTURE.md) and [specs/](specs/)
- **Issues**: [github.com/ekohan/fleetmix/issues](https://github.com/ekohan/fleetmix/issues)
- **Examples**: Check `examples/` directory

---

## Next Steps

- **Experiment**: Try parameter variations
- **Customize**: Add custom clustering (see [specs/protocols.md](specs/protocols.md))
- **Integrate**: Use Python API in your workflows
- **Validate**: Compare with your current operations

---

**Last Updated**: 2025-10-05

**Navigation**: [← Docs Home](README.md) | [Quickstart](quickstart.md) | [Architecture](ARCHITECTURE.md)

