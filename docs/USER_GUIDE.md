# FleetMix User Guide

> **Complete guide for practitioners solving real-world fleet design problems**

---

## Getting Started

### Installation

```bash
pip install fleetmix
```

Or from source:
```bash
git clone https://github.com/ekohan/fleetmix.git
cd fleetmix
./init.sh
pip install -e .
```

### Your First Optimization

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
- Example: "3 SmallTruck, 2 LargeTruck with Dry+Chilled compartments"

**Utilization**: How full vehicles are on average
- Target: 70-85% is good
- Too low (<60%): Consider smaller/fewer vehicles
- Too high (>90%): May need more vehicles or capacity

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
  mip_gap: 0.05  # 5% gap OK for large problems
```

### For Better Solution Quality

1. **Use Gurobi solver**:
```yaml
optimization:
  solver: gurobi  # Much better than CBC
```

2. **Enable improvement phase**:
```yaml
optimization:
  improvement_enabled: true
  max_improvement_iterations: 5
```

3. **Use more clustering methods**:
```yaml
clustering:
  methods: [minibatch_kmeans, kmedoids, gaussian_mixture, agglomerative]
```

4. **Try demand-aware clustering**:
```yaml
clustering:
  geo_weight: [1.0, 0.8, 0.6]
  demand_weight: [0.0, 0.2, 0.4]
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

### "Solution uses many vehicles"

**Causes**:
- Fixed costs too low relative to variable costs
- Capacity or time constraints binding
- Demand very dispersed geographically

**Solutions**:
- Increase fixed costs to penalize vehicle usage
- Relax `max_route_time` if possible
- Consider larger vehicles
- Check if `avg_speed` is realistic

### "Optimization takes too long"

**Solutions**:
- Use CBC solver initially (free), switch to Gurobi for production
- Reduce clustering variants (see "For Faster Results" above)
- Set `optimization.time_limit` to acceptable level
- For 1000+ customers, expect 30-120s (CBC) or 10-30s (Gurobi)

### "Results seem unrealistic"

**Check**:
- Units consistency (all in kg? all in km/h?)
- Cost parameters (fixed vs variable balance)
- Geographic coordinates (latitude/longitude correct?)
- Demand values (not accidentally scaled?)

---

## Best Practices

### Data Preparation

✅ **Do**:
- Clean data: remove duplicates, invalid coordinates
- Validate: all demand values non-negative
- Test small: start with subset (100 customers) before full run

❌ **Don't**:
- Mix units (some km, some miles)
- Include depot as customer
- Use address strings (must convert to lat/lon)

### Configuration

✅ **Do**:
- Start with realistic baseline parameters
- Document assumptions in config file comments
- Version control configs (track parameter changes)

❌ **Don't**:
- Set `max_route_time` impossibly low
- Use same costs for very different vehicle types
- Forget to specify `goods` list

### Interpretation

✅ **Do**:
- Compare multiple scenarios
- Validate against operational experience
- Consider qualitative factors (driver availability, contracts)

❌ **Don't**:
- Trust single run without sensitivity check
- Ignore utilization metrics
- Deploy without operational validation

---

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
- **Validate**: Compare with current operations

---

**Last Updated**: 2025-10-05

**Navigation**: [← Docs Home](README.md) | [Quickstart](quickstart.md) | [Architecture](ARCHITECTURE.md)

