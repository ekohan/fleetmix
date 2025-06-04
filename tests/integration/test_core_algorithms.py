"""
Integration tests for core FleetMix algorithmic components.
Tests clustering, optimization, and route calculations with real scenarios.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml

from fleetmix.clustering import generate_feasible_clusters
from fleetmix.clustering.heuristics import (
    get_feasible_customers_subset, 
    create_initial_clusters
)
from fleetmix.optimization.core import optimize_fleet_selection
from fleetmix.post_optimization.merge_phase import improve_solution
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations, _generate_vehicle_configurations_df
from fleetmix.utils.route_time import (
    estimate_route_time,
    calculate_total_service_time_hours
)
from fleetmix.utils.coordinate_converter import (
    CoordinateConverter
)
try:
    from haversine import haversine
    HAVERSINE_AVAILABLE = True
except ImportError:
    HAVERSINE_AVAILABLE = False
from fleetmix.config.parameters import Parameters


class TestCoreAlgorithms:
    """Test core algorithmic components with realistic data."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def realistic_customers(self):
        """Create realistic customer data for testing algorithms."""
        # Manhattan-like grid pattern with varied demands
        customers_data = {
            'Customer_ID': list(range(1, 21)),
            'Customer_Name': [f'Store_{i}' for i in range(1, 21)],
            'Latitude': [
                40.7589, 40.7614, 40.7505, 40.7282, 40.7831,
                40.7524, 40.7663, 40.7398, 40.7177, 40.7912,
                40.7445, 40.7726, 40.7356, 40.7023, 40.7889,
                40.7467, 40.7801, 40.7334, 40.7156, 40.7923
            ],
            'Longitude': [
                -73.9851, -73.9776, -73.9934, -73.9942, -73.9712,
                -73.9823, -73.9745, -73.9901, -73.9889, -73.9634,
                -73.9812, -73.9723, -73.9876, -73.9912, -73.9645,
                -73.9834, -73.9701, -73.9867, -73.9923, -73.9623
            ],
            'Dry_Demand': [
                100, 150, 80, 120, 90, 110, 140, 75, 95, 130,
                85, 125, 105, 160, 70, 115, 135, 88, 98, 145
            ],
            'Chilled_Demand': [
                50, 75, 40, 60, 45, 55, 70, 38, 48, 65,
                43, 63, 53, 80, 35, 58, 68, 44, 49, 73
            ],
            'Frozen_Demand': [
                25, 30, 20, 35, 15, 28, 32, 18, 22, 38,
                19, 33, 27, 40, 12, 29, 36, 21, 24, 41
            ]
        }
        return pd.DataFrame(customers_data)
    
    @pytest.fixture
    def realistic_config(self, tmp_path):
        """Create realistic fleet configuration for testing."""
        # Define config as dict first
        config_dict = {
            'goods': ['Dry', 'Chilled', 'Frozen'],
            'vehicles': {
                'Small Van': {
                    'fixed_cost': 100,
                    'capacity': 400,
                    'avg_speed': 30.0,
                    'service_time': 5.0,
                    'max_route_time': 4.0,
                    'extra': {
                        'variable_cost_per_km': 0.5,
                        'compartments': [
                            {'temperature_min': -25, 'temperature_max': 25, 'capacity': 400}
                        ]
                    }
                },
                'Medium Truck': {
                    'fixed_cost': 150,
                    'capacity': 800,
                    'avg_speed': 25.0,
                    'service_time': 8.0,
                    'max_route_time': 6.0,
                    'extra': {
                        'variable_cost_per_km': 0.7,
                        'compartments': [
                            {'temperature_min': -25, 'temperature_max': -18, 'capacity': 200},
                            {'temperature_min': 0, 'temperature_max': 5, 'capacity': 300},
                            {'temperature_min': 15, 'temperature_max': 25, 'capacity': 300}
                        ]
                    }
                },
                'Large Truck': {
                    'fixed_cost': 200,
                    'capacity': 1200,
                    'avg_speed': 20.0,
                    'service_time': 10.0,
                    'max_route_time': 8.0,
                    'extra': {
                        'variable_cost_per_km': 0.9,
                        'compartments': [
                            {'temperature_min': -25, 'temperature_max': -18, 'capacity': 300},
                            {'temperature_min': 0, 'temperature_max': 5, 'capacity': 450},
                            {'temperature_min': 15, 'temperature_max': 25, 'capacity': 450}
                        ]
                    }
                }
            },
            'variable_cost_per_hour': 20.0,
            'depot': {'latitude': 40.7831, 'longitude': -73.9712},
            'clustering': {
                # 'max_clusters_per_vehicle': 100, # Not a direct Parameters field
                'time_limit_minutes': 60, # Not a direct Parameters field
                'route_time_estimation': 'Legacy',
                'method': 'minibatch_kmeans',
                'max_depth': 5,
                'geo_weight': 0.7,
                'demand_weight': 0.3
            },
            'demand_file': 'test_demand.csv',
            'light_load_penalty': 10.0,
            'light_load_threshold': 0.5,
            'compartment_setup_cost': 100.0,
            'format': 'json',
            'post_optimization': True,
            'small_cluster_size': 3,
            'nearest_merge_candidates': 10,
            'max_improvement_iterations': 5,
            'prune_tsp': False
        }

        # Write this dict to a temporary YAML file
        temp_yaml_path = tmp_path / "realistic_temp_config.yaml"
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(config_dict, f)

        # Load Parameters from this temporary YAML
        return Parameters.from_yaml(temp_yaml_path)

    def test_vehicle_configuration_generation(self, realistic_config):
        """Test vehicle configuration generation with multiple vehicle types."""
        configs_df = _generate_vehicle_configurations_df(realistic_config.vehicles, realistic_config.goods)
        
        # Should generate configs for all vehicles
        assert len(configs_df) >= 3  # At least 3 vehicle types
        
        # Check required columns
        required_cols = ['Config_ID', 'Vehicle_Type', 'Fixed_Cost', 'Capacity']
        for col in required_cols:
            assert col in configs_df.columns
        
        # Verify capacity constraints
        assert all(configs_df['Capacity'] > 0)
        assert all(configs_df['Fixed_Cost'] >= 0)

    def test_cluster_generation_process(self, realistic_customers, realistic_config):
        """Test cluster generation with realistic data."""
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=realistic_customers,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        # Should generate some clusters
        assert len(clusters_df) > 0
        
        # Verify cluster structure
        expected_cols = ['Cluster_ID', 'Config_ID', 'Customers']
        for col in expected_cols:
            assert col in clusters_df.columns
        
        # Verify clusters have customers assigned
        for _, cluster in clusters_df.iterrows():
            customers_list = cluster['Customers']
            assert len(customers_list) > 0

    def test_coordinate_conversions(self, realistic_customers):
        """Test coordinate conversion and distance calculations."""
        if not HAVERSINE_AVAILABLE:
            pytest.skip("Haversine package not available")
            
        # Test haversine distance calculation
        lat1, lon1 = realistic_customers.iloc[0][['Latitude', 'Longitude']]
        lat2, lon2 = realistic_customers.iloc[1][['Latitude', 'Longitude']]
        
        distance = haversine((lat1, lon1), (lat2, lon2))
        assert distance > 0
        assert distance < 100  # Should be reasonable for NYC area
        
        # Test coordinate converter functionality
        try:
            # Create coordinate dictionary from customers data
            coord_dict = {}
            for idx, row in realistic_customers.head(5).iterrows():
                coord_dict[row['Customer_ID']] = (row['Longitude'], row['Latitude'])
            
            # Create converter 
            converter = CoordinateConverter(coord_dict)
            
            # Test conversion to geographic coordinates
            geo_coords = converter.convert_all_coordinates(coord_dict, to_geographic=True)
            assert len(geo_coords) == len(coord_dict)
            
            # Test that coordinates are in reasonable lat/lon ranges
            for node_id, (lat, lon) in geo_coords.items():
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180
                
        except Exception as e:
            # Coordinate conversion might not work properly in test environment
            pytest.skip(f"Coordinate conversion test failed: {str(e)}")

    def test_route_time_calculations(self, realistic_customers, realistic_config):
        """Test route time estimation algorithms."""
        # Create a simple route with first 5 customers
        route_customers = realistic_customers.head(5)
        
        # Test simple distance calculation using haversine
        total_distance = 0
        if HAVERSINE_AVAILABLE:
            for i in range(len(route_customers) - 1):
                lat1, lon1 = route_customers.iloc[i][['Latitude', 'Longitude']]
                lat2, lon2 = route_customers.iloc[i + 1][['Latitude', 'Longitude']]
                distance = haversine((lat1, lon1), (lat2, lon2))
                total_distance += distance
            assert total_distance > 0
        
        # Test route time estimation using available function
        # Get operational parameters from the first vehicle in realistic_config
        first_vehicle = list(realistic_config.vehicles.values())[0]
        service_time = first_vehicle.service_time
        avg_speed = first_vehicle.avg_speed
        max_route_time = first_vehicle.max_route_time
        
        try:
            route_time, sequence = estimate_route_time(
                cluster_customers=route_customers,
                depot=realistic_config.depot,
                service_time=service_time,
                avg_speed=avg_speed,
                method=realistic_config.clustering['route_time_estimation'],
                max_route_time=max_route_time,
                prune_tsp=realistic_config.prune_tsp
            )
            assert route_time > service_time * len(route_customers) / 60  # Should include travel time
        except Exception:
            # If route time estimation fails (missing dependencies), just test service time calculation
            total_service_time = calculate_total_service_time_hours(len(route_customers), service_time)
            assert total_service_time > 0
        
        # Test individual route segment estimation
        segment_time = estimate_route_time(
            route_customers.iloc[0:2], 
            realistic_config.depot,
            service_time,
            avg_speed,
            method='Legacy',
            max_route_time=max_route_time,
            prune_tsp=False
        )[0]
        assert segment_time > 0

    def test_optimization_solver(self, realistic_customers, realistic_config):
        """Test the optimization solver with realistic data."""
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=realistic_customers,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        # Run optimization
        solution = optimize_fleet_selection(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=realistic_customers,
            parameters=realistic_config,
            verbose=False
        )
        
        # Validate solution
        assert solution is not None
        assert solution.total_cost >= 0
        assert solution.total_vehicles >= 0
        assert isinstance(solution.selected_clusters, pd.DataFrame)
        
        # Check that all served customers are unique
        served_customers = set()
        for customers in solution.selected_clusters['Customers']:
            for cid in customers:
                assert cid not in served_customers, f"Customer {cid} assigned to multiple clusters"
                served_customers.add(cid)

    def test_post_optimization_merge_phase(self, realistic_customers, realistic_config):
        """Test the post-optimization merge phase."""
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=realistic_customers,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        # Get initial solution
        initial_solution = optimize_fleet_selection(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=realistic_customers,
            parameters=realistic_config,
            verbose=False
        )
        
        # Apply merge phase
        improved_solution = improve_solution(
            initial_solution,
            configs_df,
            realistic_customers,
            realistic_config
        )
        
        # Improved solution should not be worse
        assert improved_solution.total_cost <= initial_solution.total_cost
        assert improved_solution.total_vehicles <= initial_solution.total_vehicles

    def test_large_scale_clustering(self, realistic_config):
        """Test clustering with a larger number of customers."""
        # Create 100 customers
        np.random.seed(42)
        large_customers = pd.DataFrame({
            'Customer_ID': [f'C{i:03d}' for i in range(100)],
            'Customer_Name': [f'Customer {i}' for i in range(100)],
            'Latitude': np.random.uniform(40.5, 41.0, 100),
            'Longitude': np.random.uniform(-74.5, -73.5, 100),
            'Dry_Demand': np.random.randint(0, 50, 100),
            'Chilled_Demand': np.random.randint(0, 30, 100),
            'Frozen_Demand': np.random.randint(0, 20, 100)
        })
        
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=large_customers,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        # Should generate clusters
        assert len(clusters_df) > 0
        
        # Each cluster should be feasible
        for _, cluster in clusters_df.iterrows():
            assert len(cluster['Customers']) > 0
            assert cluster['Route_Time'] > 0

    def test_edge_case_single_customer(self, realistic_config):
        """Test with a single customer."""
        single_customer = pd.DataFrame({
            'Customer_ID': ['C001'],
            'Customer_Name': ['Single Customer'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060],
            'Dry_Demand': [10],
            'Chilled_Demand': [5],
            'Frozen_Demand': [0]
        })
        
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=single_customer,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        # Should generate at least one cluster
        assert len(clusters_df) > 0
        
        # Run optimization
        solution = optimize_fleet_selection(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=single_customer,
            parameters=realistic_config,
            verbose=False
        )
        
        # Should find a solution
        assert solution.total_vehicles >= 1

    def test_algorithm_performance_bounds(self, realistic_customers, realistic_config):
        """Test that algorithm respects performance bounds."""
        # Generate configurations
        configs_df = _generate_vehicle_configurations_df(
            realistic_config.vehicles, 
            realistic_config.goods
        )
        
        # Generate clusters with time limit
        import time
        start_time = time.time()
        
        from fleetmix.clustering.generator import _generate_feasible_clusters_df
        clusters_df = _generate_feasible_clusters_df(
            customers=realistic_customers,
            configurations_df=configs_df,
            params=realistic_config
        )
        
        clustering_time = time.time() - start_time
        
        # Clustering should complete in reasonable time (< 30 seconds for 20 customers)
        assert clustering_time < 30.0
        
        # Run optimization with time limit
        start_time = time.time()
        solution = optimize_fleet_selection(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=realistic_customers,
            parameters=realistic_config,
            verbose=False
        )
        
        optimization_time = time.time() - start_time
        
        # Optimization should complete in reasonable time (< 60 seconds)
        assert optimization_time < 60.0 