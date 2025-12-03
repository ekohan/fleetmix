# FleetMix User Guide

> **Complete guide for practitioners solving real-world fleet design problems**

---

## Getting Started

See [quickstart.md](quickstart.md) for installation instructions.

### Your First Fleet Size & Mix (FSM) Optimization

**Step 1**: Prepare your customer demand data (CSV format)

```csv
Customer_ID,Latitude,Longitude,Dry_Demand,Chilled_Demand,Frozen_Demand
C001,4.6097,-74.0817,150,80,40
C002,4.6512,-74.1120,200,100,50
C003,4.7234,-74.0654,100,60,30
```

> **Note**: FleetMix also accepts long-format CSVs (`ClientID,Lat,Lon,Kg,ProductType`) which are auto-converted internally.

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
  --format xlsx
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
  MultiProductTruck:
    # No allowed_goods = can carry all product types
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
    customers = pd.read_csv("customers.csv")
    demand_cols = ['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']
    customers[demand_cols] *= demand_multiplier
    
    # Optimize
    solution = fm.optimize(demand=customers, config="config.yaml")
    
    results.append({
        'demand_level': demand_multiplier,
        'vehicles_needed': solution.total_vehicles,
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
import fleetmix as fm
from fleetmix.config import load_fleetmix_params
import dataclasses

for max_time in [6, 7, 8, 9, 10, 11, 12]:
    params = load_fleetmix_params("config.yaml")
    
    # Update max_route_time for all vehicles
    updated_vehicles = {
        name: dataclasses.replace(spec, max_route_time=max_time)
        for name, spec in params.problem.vehicles.items()
    }
    params = dataclasses.replace(
        params, 
        problem=dataclasses.replace(params.problem, vehicles=updated_vehicles)
    )
    
    solution = fm.optimize(demand="customers.csv", config=params)
    print(f"{max_time}h: {solution.total_vehicles} vehicles")
```

**Expected**: Tighter time limits require more vehicles

---

## Tuning Performance

### For Faster Results (Large Problems)

1. **Use fewer clustering methods**:
```yaml
clustering:
  method: minibatch_kmeans  # Fastest
```

2. **Use BHH route time (default)**:
```yaml
clustering:
  route_time_estimation: 'BHH'  # Much faster than 'TSP'
```

3. **Set solver time limit**:
```yaml
time_limit: 60  # Acceptable for approximate solution
gap_rel: 0.005  # 0.5% gap OK for large problems
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

**Navigation**: [← Docs Home](README.md) | [Quickstart](quickstart.md) | [Architecture](ARCHITECTURE.md)
