# CLI equivalent: fleetmix optimize --demand customers.csv --config fleet.yaml
# Or programmatically via the Python API:

import fleetmix as fm

solution = fm.optimize(
    demand="customers.csv",
    config="fleet_config.yaml",  # YAML file or FleetmixParams object
)

print(f"Total vehicles: {solution.total_vehicles}")
print(f"Total cost: ${solution.total_cost:,.2f}")
print(f"Vehicles by type: {solution.vehicles_used}")

for cluster in solution.selected_clusters:
    print(f"  {cluster.vehicle_type}: "
          f"{len(cluster.customers)} customers, "
          f"demand: {cluster.total_demand}")
