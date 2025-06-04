"""
Visualize Heterogeneous Fleet Characteristics
=============================================

This script creates a visual comparison of the different vehicle types
and their operational parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Vehicle data
vehicles = {
    'Bike': {
        'capacity': 50,
        'fixed_cost': 30,
        'avg_speed': 15,
        'service_time': 5,
        'max_route_time': 2,
        'goods_types': 2  # Dry + Chilled
    },
    'Truck': {
        'capacity': 500,
        'fixed_cost': 150,
        'avg_speed': 25,
        'service_time': 15,
        'max_route_time': 8,
        'goods_types': 3  # Dry + Chilled + Frozen
    },
    'Drone': {
        'capacity': 5,
        'fixed_cost': 20,
        'avg_speed': 40,
        'service_time': 2,
        'max_route_time': 0.5,
        'goods_types': 1  # Dry only
    }
}

# Create DataFrame for easier plotting
df = pd.DataFrame(vehicles).T

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Heterogeneous Fleet Comparison', fontsize=16, fontweight='bold')

# 1. Capacity Comparison
ax = axes[0, 0]
bars = ax.bar(df.index, df['capacity'], color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Capacity (kg)')
ax.set_title('Vehicle Capacity')
ax.set_ylim(0, 600)
for bar, val in zip(bars, df['capacity']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
            f'{val}kg', ha='center', va='bottom')

# 2. Speed Comparison
ax = axes[0, 1]
bars = ax.bar(df.index, df['avg_speed'], color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Speed (km/h)')
ax.set_title('Average Speed')
ax.set_ylim(0, 50)
for bar, val in zip(bars, df['avg_speed']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val}', ha='center', va='bottom')

# 3. Service Time
ax = axes[0, 2]
bars = ax.bar(df.index, df['service_time'], color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Time (minutes)')
ax.set_title('Service Time per Stop')
ax.set_ylim(0, 20)
for bar, val in zip(bars, df['service_time']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val}min', ha='center', va='bottom')

# 4. Max Route Time
ax = axes[1, 0]
bars = ax.bar(df.index, df['max_route_time'], color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Time (hours)')
ax.set_title('Maximum Route Time')
ax.set_ylim(0, 10)
for bar, val in zip(bars, df['max_route_time']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
            f'{val}h', ha='center', va='bottom')

# 5. Fixed Cost
ax = axes[1, 1]
bars = ax.bar(df.index, df['fixed_cost'], color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Cost ($)')
ax.set_title('Fixed Cost per Day')
ax.set_ylim(0, 180)
for bar, val in zip(bars, df['fixed_cost']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
            f'${val}', ha='center', va='bottom')

# 6. Goods Type Capability
ax = axes[1, 2]
goods_matrix = np.array([
    [1, 1, 0],  # Bike: Dry, Chilled
    [1, 1, 1],  # Truck: All
    [1, 0, 0]   # Drone: Dry only
])
im = ax.imshow(goods_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Dry', 'Chilled', 'Frozen'])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Bike', 'Truck', 'Drone'])
ax.set_title('Goods Type Capability')

# Add text annotations
for i in range(3):
    for j in range(3):
        text = '✓' if goods_matrix[i, j] == 1 else '✗'
        ax.text(j, i, text, ha='center', va='center', 
                color='white' if goods_matrix[i, j] == 1 else 'black',
                fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('fleet_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
print("\n📊 Fleet Characteristics Summary")
print("=" * 80)
print(f"{'Vehicle':<10} {'Capacity':<10} {'Speed':<10} {'Service':<10} {'Max Route':<12} {'Cost/Day':<10}")
print(f"{'Type':<10} {'(kg)':<10} {'(km/h)':<10} {'(min)':<10} {'(hours)':<12} {'($)':<10}")
print("-" * 80)
for vehicle, specs in vehicles.items():
    print(f"{vehicle:<10} {specs['capacity']:<10} {specs['avg_speed']:<10} "
          f"{specs['service_time']:<10} {specs['max_route_time']:<12.1f} {specs['fixed_cost']:<10}")

# Calculate operational ranges
print("\n🚀 Operational Range Analysis")
print("=" * 80)
for vehicle, specs in vehicles.items():
    # Maximum distance = speed × max_route_time - (service_time × estimated_stops)
    estimated_stops = specs['max_route_time'] * 60 / (specs['service_time'] + 10)  # 10 min travel between stops
    travel_time = specs['max_route_time'] - (estimated_stops * specs['service_time'] / 60)
    max_distance = specs['avg_speed'] * travel_time
    
    print(f"\n{vehicle}:")
    print(f"  Max distance per route: {max_distance:.1f} km")
    print(f"  Estimated stops per route: {estimated_stops:.0f}")
    print(f"  Delivery capacity per route: {specs['capacity']:.0f} kg") 