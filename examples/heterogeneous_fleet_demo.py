"""
Heterogeneous Fleet Demo
========================

This example demonstrates how different vehicle types (bikes, trucks, drones)
can have different operational parameters in the fleetmix optimizer.

Vehicle Types:
- Bike: Medium speed, 2 goods (Dry, Chilled), short routes
- Truck: Slow speed, 3 goods (Dry, Chilled, Frozen), long routes  
- Drone: Fast speed, 1 good (Dry only), very short routes
"""

import pandas as pd
from pathlib import Path
import yaml
import fleetmix

# Create sample customer data for urban delivery
customers_data = {
    'Customer_ID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'],
    'Customer_Name': [
        'Corner Store', 'Restaurant A', 'Pharmacy', 'Grocery B', 'Cafe C',
        'Market D', 'Deli E', 'Bakery F', 'Shop G', 'Office H'
    ],
    'Latitude': [
        40.7589, 40.7614, 40.7505, 40.7282, 40.7831,
        40.7524, 40.7663, 40.7398, 40.7177, 40.7612
    ],
    'Longitude': [
        -73.9851, -73.9776, -73.9934, -73.9942, -73.9712,
        -73.9823, -73.9745, -73.9901, -73.9889, -73.9798
    ],
    'Dry_Demand': [20, 15, 30, 0, 25, 40, 10, 35, 15, 20],      # Light packages, documents
    'Chilled_Demand': [10, 25, 0, 30, 15, 20, 35, 0, 20, 0],    # Food, medicine
    'Frozen_Demand': [0, 15, 0, 20, 0, 10, 25, 0, 0, 0]          # Frozen food
}

customers_df = pd.DataFrame(customers_data)

# Create configuration with heterogeneous fleet
config = {
    'goods': ['Dry', 'Chilled', 'Frozen'],
    
    'vehicles': {
        'Bike': {
            'capacity': 50,           # 50 kg capacity
            'fixed_cost': 30,         # Low fixed cost
            'avg_speed': 15.0,        # 15 km/h in urban traffic
            'service_time': 5.0,      # 5 minutes per stop (quick)
            'max_route_time': 2.0,    # 2 hour shifts
            'compartments': {
                'Dry': True,
                'Chilled': True,      # Insulated box
                'Frozen': False       # No freezer capability
            }
        },
        'Truck': {
            'capacity': 500,          # 500 kg capacity
            'fixed_cost': 150,        # High fixed cost
            'avg_speed': 25.0,        # 25 km/h in urban traffic
            'service_time': 15.0,     # 15 minutes per stop (loading/unloading)
            'max_route_time': 8.0,    # Full day shifts
            'compartments': {
                'Dry': True,
                'Chilled': True,
                'Frozen': True        # Full refrigeration
            }
        },
        'Drone': {
            'capacity': 5,            # 5 kg capacity
            'fixed_cost': 20,         # Low fixed cost
            'avg_speed': 40.0,        # 40 km/h (direct flight)
            'service_time': 2.0,      # 2 minutes per stop (drop-off)
            'max_route_time': 0.5,    # 30 minute battery life
            'compartments': {
                'Dry': True,          # Small packages only
                'Chilled': False,
                'Frozen': False
            }
        }
    },
    
    'depot': {
        'latitude': 40.7580,
        'longitude': -73.9855
    },
    
    'variable_cost_per_hour': 25.0,
    'light_load_penalty': 10.0,
    'light_load_threshold': 0.3,
    'compartment_setup_cost': 50.0,
    
    'clustering': {
        'method': 'combine',
        'route_time_estimation': 'BHH',
        'max_depth': 20,
        'geo_weight': 0.8,        # High geographic weight for urban delivery
        'demand_weight': 0.2
    },
    
    'post_optimization': True,
    'small_cluster_size': 2,
    'nearest_merge_candidates': 5,
    'max_improvement_iterations': 3
}

def main():
    """Run the heterogeneous fleet optimization."""
    print("🚴 🚚 🚁 Heterogeneous Fleet Optimization Demo")
    print("=" * 60)
    
    # Save customer data
    customers_df.to_csv('urban_customers.csv', index=False)
    print(f"✓ Created {len(customers_df)} urban delivery customers")
    
    # Save configuration
    with open('heterogeneous_fleet_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("✓ Created fleet configuration with bikes, trucks, and drones")
    
    # Display fleet characteristics
    print("\n📊 Fleet Characteristics:")
    print("-" * 60)
    print(f"{'Vehicle':<10} {'Capacity':<10} {'Speed':<10} {'Service':<10} {'Max Route':<10}")
    print(f"{'Type':<10} {'(kg)':<10} {'(km/h)':<10} {'(min)':<10} {'(hours)':<10}")
    print("-" * 60)
    for vehicle_type, specs in config['vehicles'].items():
        print(f"{vehicle_type:<10} {specs['capacity']:<10} {specs['avg_speed']:<10} "
              f"{specs['service_time']:<10} {specs['max_route_time']:<10}")
    
    # Display demand summary
    print("\n📦 Demand Summary:")
    print(f"Total Dry Demand: {customers_df['Dry_Demand'].sum()} kg")
    print(f"Total Chilled Demand: {customers_df['Chilled_Demand'].sum()} kg")
    print(f"Total Frozen Demand: {customers_df['Frozen_Demand'].sum()} kg")
    
    # Run optimization
    print("\n🔄 Running optimization...")
    solution = fleetmix.optimize(
        demand='urban_customers.csv',
        config='heterogeneous_fleet_config.yaml',
        output_dir='results',
        format='excel',
        verbose=True
    )
    
    # Display results
    print("\n✅ Optimization Results:")
    print("-" * 60)
    print(f"Total Cost: ${solution.total_cost:,.2f}")
    print(f"Fixed Cost: ${solution.total_fixed_cost:,.2f}")
    print(f"Variable Cost: ${solution.total_variable_cost:,.2f}")
    print(f"Penalties: ${solution.total_penalties:,.2f}")
    
    print(f"\nVehicles Used:")
    for vehicle_type, count in solution.vehicles_used.items():
        print(f"  {vehicle_type}: {count}")
    
    # Analyze vehicle utilization by type
    if hasattr(solution, 'selected_clusters') and not solution.selected_clusters.empty:
        print("\n📈 Vehicle Type Analysis:")
        print("-" * 60)
        
        # Group by vehicle type
        for vehicle_type in config['vehicles'].keys():
            vehicle_clusters = solution.selected_clusters[
                solution.selected_clusters['Config_ID'].str.contains(vehicle_type)
            ]
            
            if not vehicle_clusters.empty:
                avg_customers = vehicle_clusters['Customers'].apply(len).mean()
                avg_route_time = vehicle_clusters['Route_Time'].mean()
                total_deliveries = sum(len(c) for c in vehicle_clusters['Customers'])
                
                print(f"\n{vehicle_type}:")
                print(f"  Routes: {len(vehicle_clusters)}")
                print(f"  Avg customers/route: {avg_customers:.1f}")
                print(f"  Avg route time: {avg_route_time:.1f} hours")
                print(f"  Total deliveries: {total_deliveries}")
    
    print("\n✨ Optimization complete! Results saved to 'results' directory.")
    
    # Clean up temporary files
    Path('urban_customers.csv').unlink(missing_ok=True)
    Path('heterogeneous_fleet_config.yaml').unlink(missing_ok=True)


if __name__ == "__main__":
    main() 