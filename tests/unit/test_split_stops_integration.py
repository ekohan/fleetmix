"""
Integration tests for split-stop functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fleetmix.api import optimize
from fleetmix.config.parameters import Parameters


class TestSplitStopsIntegration:
    """Integration tests for split-stop functionality."""

    @pytest.fixture
    def sample_demand_data(self):
        """Create sample demand data for testing."""
        return pd.DataFrame(
            [
                {
                    "Customer_ID": "C001",
                    "Latitude": 40.7831,
                    "Longitude": -73.9712,
                    "Dry_Demand": 100,
                    "Chilled_Demand": 50,
                    "Frozen_Demand": 0,
                },
                {
                    "Customer_ID": "C002",
                    "Latitude": 40.7589,
                    "Longitude": -73.9851,
                    "Dry_Demand": 0,
                    "Chilled_Demand": 0,
                    "Frozen_Demand": 75,
                },
                {
                    "Customer_ID": "C003",
                    "Latitude": 40.7505,
                    "Longitude": -73.9934,
                    "Dry_Demand": 25,
                    "Chilled_Demand": 25,
                    "Frozen_Demand": 25,
                },
            ]
        )

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing."""
        import yaml

        config_data = {
            "vehicles": {
                "SmallTruck": {
                    "capacity": 100,
                    "fixed_cost": 100,
                    "avg_speed": 30,
                    "service_time": 25,
                    "max_route_time": 24,  # in hours
                    "compartments": {"Dry": True, "Chilled": True, "Frozen": True},
                },
                "LargeTruck": {
                    "capacity": 200,
                    "fixed_cost": 150,
                    "avg_speed": 30,
                    "service_time": 25,
                    "max_route_time": 24,  # in hours
                    "compartments": {"Dry": True, "Chilled": True, "Frozen": True},
                },
            },
            "depot": {"latitude": 40.7831, "longitude": -73.9712},
            "goods": ["Dry", "Chilled", "Frozen"],
            "variable_cost_per_hour": 20.0,
            "clustering": {
                "max_depth": 5,
                "method": "minibatch_kmeans",
                "route_time_estimation": "BHH",
                "geo_weight": 0.7,
                "demand_weight": 0.3,
                "distance": "euclidean",
            },
            "demand_file": "test_data.csv",  # Required field
            "light_load_penalty": 0,
            "light_load_threshold": 0,
            "compartment_setup_cost": 0,
            "post_optimization": False,
            "allow_split_stops": False,  # Default value
            "format": "json",
        }

        # Create a temporary YAML file and use Parameters.from_yaml()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name

        try:
            params = Parameters.from_yaml(temp_config_path)
            return params
        finally:
            Path(temp_config_path).unlink()  # Clean up temp file

    def test_single_stop_mode(self, sample_demand_data, minimal_config):
        """Test optimization with split-stops disabled (default behavior)."""
        # Ensure split-stops is disabled
        minimal_config.allow_split_stops = False

        # Run optimization
        solution = optimize(
            demand=sample_demand_data,
            config=minimal_config,
            output_dir="",  # Don't save results
            verbose=False,
            allow_split_stops=False,
        )

        # Verify solution is valid
        assert solution.total_cost > 0
        assert solution.total_vehicles > 0
        assert len(solution.missing_customers) == 0

        # In single-stop mode, each physical customer should appear at most once
        served_customers = set()
        for cluster in solution.selected_clusters:
            for customer_id in cluster.customers:
                # Should not contain pseudo-customer IDs (no '::')
                assert "::" not in customer_id
                served_customers.add(customer_id)

        # All original customers should be served
        original_customers = set(sample_demand_data["Customer_ID"])
        assert served_customers == original_customers

    def test_split_stop_mode(self, sample_demand_data, minimal_config):
        """Test optimization with split-stops enabled."""
        # Enable split-stops
        minimal_config.allow_split_stops = True

        # Run optimization
        solution = optimize(
            demand=sample_demand_data,
            config=minimal_config,
            output_dir="",  # Don't save results
            verbose=False,
            allow_split_stops=True,
        )

        # Verify solution is valid
        assert solution.total_cost > 0
        assert solution.total_vehicles > 0
        # Note: missing_customers may be > 0 in split-stop mode as it counts unselected pseudo-customers

        # In split-stop mode, we should see pseudo-customer IDs
        served_pseudo_customers = set()
        has_pseudo_customers = False

        for cluster in solution.selected_clusters:
            for customer_id in cluster.customers:
                served_pseudo_customers.add(customer_id)
                if "::" in customer_id:
                    has_pseudo_customers = True

        # Should have pseudo-customers since we have customers with multiple goods
        assert has_pseudo_customers

        # Verify that each physical customer's goods are properly covered
        # Import helper functions to validate coverage
        from fleetmix.preprocess.demand import get_origin_id, get_subset_from_id

        # Build coverage map: physical_customer -> set of goods covered
        coverage_map = {}
        for customer_id in served_pseudo_customers:
            if "::" in customer_id:  # It's a pseudo-customer
                origin = get_origin_id(customer_id)
                goods_subset = get_subset_from_id(customer_id)
                if origin not in coverage_map:
                    coverage_map[origin] = set()
                coverage_map[origin].update(goods_subset)
            else:  # Direct customer (single good type)
                # For direct customers, we need to check what goods they actually have
                customer_row = sample_demand_data[
                    sample_demand_data["Customer_ID"] == customer_id
                ].iloc[0]
                goods_needed = set()
                for good in ["Dry", "Chilled", "Frozen"]:
                    if customer_row[f"{good}_Demand"] > 0:
                        goods_needed.add(good)
                coverage_map[customer_id] = goods_needed

        # Verify each physical customer has all their goods covered
        for _, customer_row in sample_demand_data.iterrows():
            customer_id = customer_row["Customer_ID"]
            required_goods = set()
            for good in ["Dry", "Chilled", "Frozen"]:
                if customer_row[f"{good}_Demand"] > 0:
                    required_goods.add(good)

            # Check coverage
            covered_goods = coverage_map.get(customer_id, set())
            assert required_goods.issubset(covered_goods), (
                f"Customer {customer_id} missing goods: {required_goods - covered_goods}"
            )

    def test_backward_compatibility(self, sample_demand_data, minimal_config):
        """Test that not specifying allow_split_stops maintains backward compatibility."""
        # Don't set allow_split_stops explicitly - should default to False
        solution1 = optimize(
            demand=sample_demand_data,
            config=minimal_config,
            output_dir="",
            verbose=False,
            # allow_split_stops not specified - should default to False
        )

        # Explicitly set to False
        solution2 = optimize(
            demand=sample_demand_data,
            config=minimal_config,
            output_dir="",
            verbose=False,
            allow_split_stops=False,
        )

        # Both solutions should be identical (same cost, same vehicle count)
        assert solution1.total_cost == solution2.total_cost
        assert solution1.total_vehicles == solution2.total_vehicles

        # Neither should have pseudo-customers
        for solution in [solution1, solution2]:
            for cluster in solution.selected_clusters:
                for customer_id in cluster.customers:
                    assert "::" not in customer_id
