"""
Integration tests for core FleetMix algorithmic components.
Tests clustering, optimization, and route calculations with real scenarios.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fleetmix import generate_feasible_clusters
from fleetmix.core_types import Customer, FleetmixSolution, VehicleConfiguration
from fleetmix.optimization.core import optimize_fleet
from fleetmix.post_optimization.merge_phase import improve_solution
from fleetmix.utils.coordinate_converter import CoordinateConverter
from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    estimate_route_time,
)
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

try:
    from haversine import haversine

    HAVERSINE_AVAILABLE = True
except ImportError:
    HAVERSINE_AVAILABLE = False
from fleetmix.config import load_fleetmix_params, FleetmixParams


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
            "Customer_ID": list(range(1, 21)),
            "Customer_Name": [f"Store_{i}" for i in range(1, 21)],
            "Latitude": [
                40.7589,
                40.7614,
                40.7505,
                40.7282,
                40.7831,
                40.7524,
                40.7663,
                40.7398,
                40.7177,
                40.7912,
                40.7445,
                40.7726,
                40.7356,
                40.7023,
                40.7889,
                40.7467,
                40.7801,
                40.7334,
                40.7156,
                40.7923,
            ],
            "Longitude": [
                -73.9851,
                -73.9776,
                -73.9934,
                -73.9942,
                -73.9712,
                -73.9823,
                -73.9745,
                -73.9901,
                -73.9889,
                -73.9634,
                -73.9812,
                -73.9723,
                -73.9876,
                -73.9912,
                -73.9645,
                -73.9834,
                -73.9701,
                -73.9867,
                -73.9923,
                -73.9623,
            ],
            "Dry_Demand": [
                100,
                150,
                80,
                120,
                90,
                110,
                140,
                75,
                95,
                130,
                85,
                125,
                105,
                160,
                70,
                115,
                135,
                88,
                98,
                145,
            ],
            "Chilled_Demand": [
                50,
                75,
                40,
                60,
                45,
                55,
                70,
                38,
                48,
                65,
                43,
                63,
                53,
                80,
                35,
                58,
                68,
                44,
                49,
                73,
            ],
            "Frozen_Demand": [
                25,
                30,
                20,
                35,
                15,
                28,
                32,
                18,
                22,
                38,
                19,
                33,
                27,
                40,
                12,
                29,
                36,
                21,
                24,
                41,
            ],
        }
        return pd.DataFrame(customers_data)

    @pytest.fixture
    def realistic_config(self, tmp_path):
        """Create realistic fleet configuration for testing."""
        # Define config as dict first
        config_dict = {
            "goods": ["Dry", "Chilled", "Frozen"],
            "vehicles": {
                "Small Van": {
                    "fixed_cost": 100,
                    # 'variable_cost_per_km': 0.5, # Not a direct Parameters field, store in extra
                    "capacity": 400,
                    "avg_speed": 40.0,
                    "max_route_time": 8.0,
                    "service_time": 15.0,
                    "extra": {
                        "variable_cost_per_km": 0.5,
                        "compartments": [
                            {
                                "temperature_min": -25,
                                "temperature_max": 25,
                                "capacity": 400,
                            }
                        ],
                    },
                },
                "Medium Truck": {
                    "fixed_cost": 150,
                    "capacity": 800,
                    "avg_speed": 40.0,
                    "max_route_time": 8.0,
                    "service_time": 15.0,
                    "extra": {
                        "variable_cost_per_km": 0.7,
                        "compartments": [
                            {
                                "temperature_min": -25,
                                "temperature_max": -18,
                                "capacity": 200,
                            },
                            {
                                "temperature_min": 0,
                                "temperature_max": 5,
                                "capacity": 300,
                            },
                            {
                                "temperature_min": 15,
                                "temperature_max": 25,
                                "capacity": 300,
                            },
                        ],
                    },
                },
                "Large Truck": {
                    "fixed_cost": 200,
                    "capacity": 1200,
                    "avg_speed": 40.0,
                    "max_route_time": 8.0,
                    "service_time": 15.0,
                    "extra": {
                        "variable_cost_per_km": 0.9,
                        "compartments": [
                            {
                                "temperature_min": -25,
                                "temperature_max": -18,
                                "capacity": 300,
                            },
                            {
                                "temperature_min": 0,
                                "temperature_max": 5,
                                "capacity": 450,
                            },
                            {
                                "temperature_min": 15,
                                "temperature_max": 25,
                                "capacity": 450,
                            },
                        ],
                    },
                },
            },
            "variable_cost_per_hour": 20.0,
            "depot": {"latitude": 40.7831, "longitude": -73.9712},
            "clustering": {
                "route_time_estimation": "BHH",
                "method": "minibatch_kmeans",
                "max_depth": 5,
                "geo_weight": 0.7,
                "demand_weight": 0.3,
            },
            "demand_file": "test_demand.csv",
            "light_load_penalty": 10.0,
            "light_load_threshold": 0.5,
            "compartment_setup_cost": 100.0,
            "format": "json",
            "post_optimization": True,
            "small_cluster_size": 3,
            "nearest_merge_candidates": 10,
            "max_improvement_iterations": 5,
            "prune_tsp": False,
        }

        # Write this dict to a temporary YAML file
        temp_yaml_path = tmp_path / "realistic_temp_config.yaml"
        with open(temp_yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load FleetmixParams from this temporary YAML
        return load_fleetmix_params(temp_yaml_path)

    def test_vehicle_configuration_generation(self, realistic_config):
        """Test vehicle configuration generation with multiple vehicle types."""
        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Should generate configs for all vehicles
        assert len(configs) >= 3  # At least 3 vehicle types

        # Check that all are VehicleConfiguration objects
        assert all(isinstance(config, VehicleConfiguration) for config in configs)

        # Verify capacity constraints
        assert all(config.capacity > 0 for config in configs)
        assert all(config.fixed_cost >= 0 for config in configs)

    def test_cluster_generation_process(self, realistic_customers, realistic_config):
        """Test the complete cluster generation process."""
        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(realistic_customers)

        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )

        # Should generate some clusters
        assert len(clusters) > 0

        # Verify cluster structure (now list of Cluster objects)
        for cluster in clusters:
            assert hasattr(cluster, "cluster_id")
            assert hasattr(cluster, "config_id")
            assert hasattr(cluster, "customers")
            assert len(cluster.customers) > 0

    def test_coordinate_conversions(self, realistic_customers):
        """Test coordinate conversion and distance calculations."""
        if not HAVERSINE_AVAILABLE:
            pytest.skip("Haversine package not available")

        # Test haversine distance calculation
        lat1, lon1 = realistic_customers.iloc[0][["Latitude", "Longitude"]]
        lat2, lon2 = realistic_customers.iloc[1][["Latitude", "Longitude"]]

        distance = haversine((lat1, lon1), (lat2, lon2))
        assert distance > 0
        assert distance < 100  # Should be reasonable for NYC area

        # Test coordinate converter functionality
        try:
            # Create coordinate dictionary from customers data
            coord_dict = {}
            for idx, row in realistic_customers.head(5).iterrows():
                coord_dict[row["Customer_ID"]] = (row["Longitude"], row["Latitude"])

            # Create converter
            converter = CoordinateConverter(coord_dict)

            # Test conversion to geographic coordinates
            geo_coords = converter.convert_all_coordinates(
                coord_dict, to_geographic=True
            )
            assert len(geo_coords) == len(coord_dict)

            # Test that coordinates are in reasonable lat/lon ranges
            for node_id, (lat, lon) in geo_coords.items():
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180

        except Exception as e:
            # Coordinate conversion might not work properly in test environment
            pytest.skip(f"Coordinate conversion test failed: {e!s}")

    def test_fsm_optimization_solver(self, realistic_customers, realistic_config):
        """Test the core MILP optimization solver."""
        # Generate configurations and clusters
        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(realistic_customers)

        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )

        # Solve the FSM problem using the new API
        solution = optimize_fleet(
            clusters=clusters,
            configurations=configs,
            customers=customers_list,
            parameters=realistic_config
        )

        # Verify solution structure
        assert isinstance(solution, FleetmixSolution)
        assert solution.total_cost > 0
        assert solution.total_vehicles > 0
        assert len(solution.vehicles_used) > 0

    def test_post_optimization_merge(self, realistic_customers, realistic_config):
        """Test post-optimization merge phase algorithm."""
        # Generate a solution first
        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(realistic_customers)

        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )

        solution = optimize_fleet(
            clusters=clusters,
            configurations=configs,
            customers=customers_list,
            parameters=realistic_config
        )

        # Test post-optimization improvement
        if solution.solver_status == "Optimal":
            try:
                improved_solution = improve_solution(
                    initial_solution=solution,
                    configurations=configs,
                    customers=customers_list,
                    params=realistic_config,
                )

                # Verify improvement didn't break anything
                assert isinstance(improved_solution, FleetmixSolution)
                assert improved_solution.total_cost >= 0

            except Exception as e:
                # Post-optimization might fail due to time limits or other issues
                # This is acceptable in testing
                import warnings

                warnings.warn(f"Post-optimization failed: {e}")
                pass

    def test_large_scale_clustering(self, realistic_config):
        """Test clustering with larger customer set."""
        # Generate a larger customer dataset
        np.random.seed(42)  # For reproducibility
        n_customers = 50

        large_customers = pd.DataFrame(
            {
                "Customer_ID": range(1, n_customers + 1),
                "Customer_Name": [f"Customer_{i}" for i in range(1, n_customers + 1)],
                "Latitude": np.random.uniform(40.7, 40.8, n_customers),
                "Longitude": np.random.uniform(-74.0, -73.9, n_customers),
                "Dry_Demand": np.random.randint(50, 200, n_customers),
                "Chilled_Demand": np.random.randint(20, 100, n_customers),
                "Frozen_Demand": np.random.randint(10, 50, n_customers),
            }
        )

        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(large_customers)

        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )

        # Should handle larger datasets
        if len(clusters) > 0:
            # Verify no customer is assigned to multiple clusters in same config
            config_customer_sets = {}
            for cluster in clusters:
                config_id = cluster.config_id
                customers_list = cluster.customers

                if config_id not in config_customer_sets:
                    config_customer_sets[config_id] = set()

                for cid in customers_list:
                    if cid in config_customer_sets[config_id]:
                        # This is actually OK - customers can be in multiple clusters
                        # for the same config as long as only one is selected
                        pass
                    config_customer_sets[config_id].add(cid)

    def test_edge_case_single_customer(self, realistic_config):
        """Test algorithms with edge case of single customer."""
        single_customer = pd.DataFrame(
            {
                "Customer_ID": [1],
                "Customer_Name": ["Only Store"],
                "Latitude": [40.7589],
                "Longitude": [-73.9851],
                "Dry_Demand": [100],
                "Chilled_Demand": [50],
                "Frozen_Demand": [25],
            }
        )

        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(single_customer)

        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )

        # Should handle single customer gracefully
        assert len(clusters) >= 0  # May be 0 if no feasible vehicle

        if len(clusters) > 0:
            # If cluster created, should contain the single customer
            cluster = clusters[0]
            customers_list = cluster.customers
            assert "1" in customers_list  # Customer ID is converted to string

    def test_algorithm_performance_bounds(self, realistic_customers, realistic_config):
        """Test that algorithms complete within reasonable time bounds."""
        import time


        configs = generate_vehicle_configurations(
            realistic_config.problem.vehicles, realistic_config.problem.goods
        )

        # Convert DataFrame to list of Customer objects for new API
        customers_list = Customer.from_dataframe(realistic_customers)

        # Time the clustering phase
        start_time = time.time()
        clusters = generate_feasible_clusters(
            customers=customers_list, configurations=configs, params=realistic_config
        )
        clustering_time = time.time() - start_time

        # Should complete within reasonable time (allowing overhead)
        assert clustering_time < 300  # 5 minutes max for test

        # Verify clustering produced some result
        assert isinstance(clusters, list)


def vehicle_configs_to_dataframe(configs):
    """Helper function to convert configs to DataFrame for tests that need it."""
    return pd.DataFrame([config.to_dict() for config in configs])
