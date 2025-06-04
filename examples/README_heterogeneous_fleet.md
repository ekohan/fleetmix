# Heterogeneous Fleet Example

This example demonstrates how fleetmix can optimize delivery routes for a mixed fleet of vehicles with different operational characteristics.

## Vehicle Types

### 🚴 Bike
- **Use Case**: Last-mile urban delivery for small packages
- **Capacity**: 50 kg (small cargo box)
- **Speed**: 15 km/h (can navigate through traffic)
- **Service Time**: 5 minutes (quick drop-off)
- **Max Route Time**: 2 hours (rider fatigue limit)
- **Goods**: Dry ✓, Chilled ✓ (insulated box), Frozen ✗

### 🚚 Truck  
- **Use Case**: Bulk deliveries and full-service routes
- **Capacity**: 500 kg (large cargo area)
- **Speed**: 25 km/h (slower in urban traffic)
- **Service Time**: 15 minutes (loading/unloading)
- **Max Route Time**: 8 hours (full work day)
- **Goods**: Dry ✓, Chilled ✓, Frozen ✓ (full refrigeration)

### 🚁 Drone
- **Use Case**: Urgent deliveries of small items
- **Capacity**: 5 kg (payload limit)
- **Speed**: 40 km/h (direct flight path)
- **Service Time**: 2 minutes (automated drop)
- **Max Route Time**: 30 minutes (battery constraint)
- **Goods**: Dry ✓ only (no temperature control)

## Key Features Demonstrated

1. **Per-Vehicle Operational Parameters**
   - Each vehicle type has its own speed, service time, and route duration
   - Reflects real-world constraints (e.g., drone battery life, bike rider fatigue)

2. **Compartment Capabilities**
   - Bikes can handle dry and chilled goods (insulated box)
   - Trucks have full refrigeration for all goods types
   - Drones limited to dry goods only

3. **Cost Structure**
   - Fixed costs vary by vehicle type ($20-$150/day)
   - Variable costs reflect fuel/energy consumption
   - Penalties for underutilized vehicles

## Running the Example

```bash
# Run the heterogeneous fleet demo
python heterogeneous_fleet_demo.py
```

This will:
1. Create sample urban delivery customers
2. Generate the fleet configuration
3. Run the optimization
4. Display results showing which vehicle types were selected

## Expected Results

The optimizer will typically:
- Use **drones** for urgent, small dry goods deliveries to nearby customers
- Deploy **bikes** for mixed small deliveries in dense urban areas  
- Assign **trucks** for:
  - Bulk deliveries requiring high capacity
  - Routes with frozen goods
  - Longer routes that exceed bike/drone limits

## Configuration File

See `heterogeneous_fleet_config.yaml` for the complete configuration with detailed comments explaining each parameter.

## Customization

You can modify the example to:
- Add more vehicle types (e.g., electric vans, cargo bikes)
- Adjust operational parameters based on your use case
- Include additional constraints (e.g., time windows, driver skills)
- Add environmental considerations (emissions per vehicle type) 