"""
Demo of the Fleetmix public API.

This example shows how to use the clean public API to:
1. Run the full optimization pipeline
2. Access the matheuristic stages individually
3. Work with the core types
"""

import pandas as pd
from fleetmix import (
    # Main function
    optimize,
    
    # Matheuristic stages
    generate_vehicle_configurations,
    generate_feasible_clusters,
    optimize_fleet_selection,
    improve_solution,
    
    # Core types
    VehicleConfiguration,
    ClusterAssignment,
    FleetmixSolution,
    Parameters,
)


def main():
    """Run a simple optimization example."""
    
    # Example 1: Simple optimization using the main API
    print("=== Example 1: Simple Optimization ===")
    
    # Create sample customer data
    customers = pd.DataFrame({
        'Customer_ID': ['C1', 'C2', 'C3', 'C4'],
        'Customer_Name': ['Store A', 'Store B', 'Store C', 'Store D'],
        'Latitude': [40.7128, 40.7580, 40.7489, 40.7549],
        'Longitude': [-74.0060, -73.9855, -73.9680, -73.9840],
        'Dry_Demand': [20, 15, 25, 10],
        'Chilled_Demand': [10, 0, 5, 15],
        'Frozen_Demand': [0, 5, 10, 0]
    })
    
    # Run optimization with default config
    solution = optimize(
        demand=customers,
        output_dir=None,  # Don't save results for this demo
        verbose=True
    )
    
    # Access the solution
    print(f"\nTotal cost: ${solution.total_cost:,.2f}")
    print(f"Vehicles used: {solution.total_vehicles}")
    print(f"Missing customers: {solution.missing_customers}")
    print(f"Solver status: {solution.solver_status}")
    
    # Inspect clusters
    print("\nCluster assignments:")
    for cluster in solution.selected_clusters:
        print(f"  Cluster {cluster.cluster_id}: {cluster.customer_ids}")
        print(f"    Config: {cluster.config_id}, Route time: {cluster.route_time:.2f}h")
    
    # Inspect vehicle configurations
    print("\nVehicle configurations used:")
    for config in solution.configurations_used:
        print(f"  Config {config.config_id}: {config.vehicle_type}")
        print(f"    Compartments: {config.compartments}")
        print(f"    Capacity: {config.capacity}, Cost: ${config.fixed_cost}")
    
    
    # Example 2: Using individual stages
    print("\n\n=== Example 2: Individual Matheuristic Stages ===")
    
    # Load parameters
    params = Parameters.from_yaml()
    
    # Stage 1: Generate vehicle configurations
    print("\n1. Generating vehicle configurations...")
    configs = generate_vehicle_configurations(params.vehicles, params.goods)
    print(f"   Generated {len(configs)} configurations")
    
    # Show first configuration
    if configs:
        print(f"   Example: {configs[0]}")
    
    # For stages 2-4, we need DataFrames for now (internal compatibility)
    # Convert configs back to DataFrame for internal stages
    configs_df = pd.DataFrame([
        {
            'Config_ID': c.config_id,
            'Vehicle_Type': c.vehicle_type,
            'Capacity': c.capacity,
            'Fixed_Cost': c.fixed_cost,
            **c.compartments
        }
        for c in configs
    ])
    
    # Stage 2: Generate feasible clusters
    print("\n2. Generating feasible clusters...")
    clusters = generate_feasible_clusters(
        customers=customers,
        configurations_df=configs_df,
        params=params
    )
    print(f"   Generated {len(clusters)} feasible clusters")
    
    # Show first cluster
    if clusters:
        print(f"   Example: Cluster {clusters[0].cluster_id} with {len(clusters[0].customer_ids)} customers")
    
    # For stages 3-4, we still need to use internal functions that expect DataFrames
    # This is a transitional state - eventually these will also use clean types
    from fleetmix.clustering.generator import _generate_feasible_clusters_df
    clusters_df = _generate_feasible_clusters_df(
        customers=customers,
        configurations_df=configs_df,
        params=params
    )
    
    # Stage 3: Optimize fleet selection
    print("\n3. Optimizing fleet selection...")
    initial_solution = optimize_fleet_selection(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers,
        parameters=params,
        verbose=False
    )
    print(f"   Initial cost: ${initial_solution.total_cost:,.2f}")
    
    # Stage 4: Improve solution
    print("\n4. Improving solution...")
    improved_solution = improve_solution(
        initial_solution,
        configs_df,
        customers,
        params
    )
    print(f"   Improved cost: ${improved_solution.total_cost:,.2f}")
    
    
    # Example 3: Working with types directly
    print("\n\n=== Example 3: Working with Types ===")
    
    # Create a vehicle configuration manually
    custom_config = VehicleConfiguration(
        config_id=99,
        vehicle_type="Custom Truck",
        compartments={"Dry": True, "Chilled": True, "Frozen": False},
        capacity=150,
        fixed_cost=750.0
    )
    print(f"\nCustom configuration: {custom_config}")
    
    # Create a cluster assignment manually
    custom_cluster = ClusterAssignment(
        cluster_id=1,
        config_id=99,
        customer_ids=['C1', 'C2'],
        route_time=2.5,
        total_demand={'Dry': 35, 'Chilled': 10, 'Frozen': 5},
        centroid=(40.7354, -73.9958)
    )
    print(f"\nCustom cluster: {custom_cluster}")


if __name__ == "__main__":
    main() 