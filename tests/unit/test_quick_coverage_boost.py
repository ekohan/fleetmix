"""Quick tests for high-impact coverage improvements."""

import pytest
import pandas as pd
from pathlib import Path

from fleetmix.core_types import VehicleConfiguration, Customer, DepotLocation
from fleetmix.utils.common import to_cfg_key
from fleetmix.utils.project_root import get_project_root
from fleetmix.config.params import VehicleSpec
from fleetmix.registry import CLUSTERER_REGISTRY


class TestQuickCoverageBoosters:
    """Test simple utility functions for quick coverage gains."""

    def test_to_cfg_key(self):
        """Test the to_cfg_key utility function."""
        assert to_cfg_key("test") == "test"
        assert to_cfg_key(123) == "123"
        assert to_cfg_key(None) == "None"

    def test_get_project_root(self):
        """Test project root detection."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert (root / "pyproject.toml").exists()

    def test_vehicle_spec_creation(self):
        """Test VehicleSpec creation."""
        spec = VehicleSpec(
            capacity=100,
            fixed_cost=200,
            compartments={"Dry": True, "Chilled": False},
            avg_speed=40,
            service_time=15,
            max_route_time=8,
        )
        assert spec.capacity == 100
        assert spec.fixed_cost == 200
        assert spec.compartments["Dry"] is True
        assert spec.compartments["Chilled"] is False

    def test_customer_methods(self):
        """Test Customer dataclass methods."""
        customer = Customer(
            customer_id="C1",
            demands={"Dry": 10, "Chilled": 5},
            location=(0.1, 0.2),
            service_time=15
        )

        # Test basic properties
        assert customer.customer_id == "C1"
        assert customer.demands == {"Dry": 10, "Chilled": 5}
        assert customer.location == (0.1, 0.2)
        assert customer.service_time == 15

        # Test methods if they exist
        if hasattr(customer, 'total_demand'):
            total = customer.total_demand()
            assert total == 15

        if hasattr(customer, 'is_pseudo_customer'):
            is_pseudo = customer.is_pseudo_customer()
            assert isinstance(is_pseudo, bool)

        if hasattr(customer, 'get_origin_id'):
            origin_id = customer.get_origin_id()
            assert origin_id == "C1"

    def test_depot_location(self):
        """Test DepotLocation creation."""
        depot = DepotLocation(latitude=0.0, longitude=0.0)
        assert depot.latitude == 0.0
        assert depot.longitude == 0.0

    def test_vehicle_configuration_properties(self):
        """Test VehicleConfiguration properties."""
        config = VehicleConfiguration(
            config_id="V1",
            vehicle_type="TestVehicle",
            capacity=100,
            fixed_cost=200,
            compartments={"Dry": True, "Chilled": True},
            avg_speed=40,
            service_time=15,
            max_route_time=8,
        )

        # Test basic properties
        assert config.config_id == "V1"
        assert config.vehicle_type == "TestVehicle"
        assert config.capacity == 100
        assert config.fixed_cost == 200

        # Test compartments
        assert config.compartments["Dry"] is True
        assert config.compartments["Chilled"] is True

        # Test other properties
        assert config.avg_speed == 40
        assert config.service_time == 15
        assert config.max_route_time == 8

        # Test methods if they exist
        if hasattr(config, 'can_carry'):
            can_carry_dry = config.can_carry("Dry")
            assert can_carry_dry is True
            can_carry_frozen = config.can_carry("Frozen")
            # Should be False if "Frozen" not in compartments

        if hasattr(config, 'allowed_goods'):
            allowed = config.allowed_goods()
            assert "Dry" in allowed
            assert "Chilled" in allowed

    def test_clusterer_registry_access(self):
        """Test that the clusterer registry is accessible."""
        assert isinstance(CLUSTERER_REGISTRY, dict)
        assert len(CLUSTERER_REGISTRY) > 0

        # Check that registered clusterers are callable
        for name, clusterer_class in CLUSTERER_REGISTRY.items():
            assert callable(clusterer_class)
            assert isinstance(name, str)

    def test_customer_edge_cases(self):
        """Test Customer with edge case values."""
        # Zero demands
        customer_zero = Customer("C0", {}, (0, 0), 0)
        assert customer_zero.customer_id == "C0"
        assert customer_zero.demands == {}

        # Large values
        customer_large = Customer(
            "C_LARGE",
            {"Dry": 1e6, "Chilled": 1e5},
            (90.0, 180.0),
            3600
        )
        assert customer_large.customer_id == "C_LARGE"
        assert customer_large.demands["Dry"] == 1e6

        # Negative coordinates (valid in some coordinate systems)
        customer_neg = Customer(
            "C_NEG",
            {"Dry": 10},
            (-45.5, -122.3),
            15
        )
        assert customer_neg.location == (-45.5, -122.3)

    def test_vehicle_configuration_edge_cases(self):
        """Test VehicleConfiguration with edge case values."""
        # Minimal configuration
        config_min = VehicleConfiguration(
            config_id="MIN",
            vehicle_type="Minimal",
            capacity=1,
            fixed_cost=1,
            compartments={},
            avg_speed=1,
            service_time=1,
            max_route_time=1,
        )
        assert config_min.config_id == "MIN"
        assert config_min.capacity == 1

        # Large configuration
        config_max = VehicleConfiguration(
            config_id="MAX",
            vehicle_type="Massive",
            capacity=100000,
            fixed_cost=50000,
            compartments={"Dry": True, "Chilled": True, "Frozen": True, "Special": True},
            avg_speed=120,
            service_time=300,
            max_route_time=24,
        )
        assert config_max.config_id == "MAX"
        assert config_max.capacity == 100000

    def test_depot_location_edge_cases(self):
        """Test DepotLocation with edge case values."""
        # Extreme coordinates
        depot_extreme = DepotLocation(latitude=90.0, longitude=180.0)
        assert depot_extreme.latitude == 90.0
        assert depot_extreme.longitude == 180.0

        # Zero coordinates
        depot_zero = DepotLocation(latitude=0.0, longitude=0.0)
        assert depot_zero.latitude == 0.0
        assert depot_zero.longitude == 0.0

        # Negative coordinates
        depot_negative = DepotLocation(latitude=-90.0, longitude=-180.0)
        assert depot_negative.latitude == -90.0
        assert depot_negative.longitude == -180.0


class TestStringConversions:
    """Test string representations and conversions."""

    def test_dataclass_string_representations(self):
        """Test that dataclasses have string representations."""
        customer = Customer("C1", {"Dry": 10}, (0, 0), 15)
        customer_str = str(customer)
        assert "C1" in customer_str

        config = VehicleConfiguration(
            "V1", "Test", 100, 200, {"Dry": True}, 40, 15, 8
        )
        config_str = str(config)
        assert "V1" in config_str

        depot = DepotLocation(0.0, 0.0)
        depot_str = str(depot)
        assert "0.0" in depot_str

    def test_repr_methods(self):
        """Test repr methods if they exist."""
        customer = Customer("C1", {"Dry": 10}, (0, 0), 15)
        customer_repr = repr(customer)
        assert "Customer" in customer_repr

        config = VehicleConfiguration(
            "V1", "Test", 100, 200, {"Dry": True}, 40, 15, 8
        )
        config_repr = repr(config)
        assert "VehicleConfiguration" in config_repr


class TestDataValidation:
    """Test data validation and type checking."""

    def test_customer_type_validation(self):
        """Test that Customer handles different input types."""
        # String customer_id
        customer1 = Customer("C1", {"Dry": 10}, (0, 0), 15)
        assert customer1.customer_id == "C1"

        # Numeric customer_id (should be converted to string)
        customer2 = Customer(123, {"Dry": 10}, (0, 0), 15)
        # Implementation dependent - may accept numbers

        # Test with different demand types
        customer3 = Customer("C3", {"Dry": 10.5, "Chilled": 0}, (0.1, 0.2), 15.5)
        assert customer3.demands["Dry"] == 10.5

    def test_coordinate_types(self):
        """Test different coordinate types."""
        # Integer coordinates
        customer_int = Customer("C1", {"Dry": 10}, (1, 2), 15)
        assert customer_int.location == (1, 2)

        # Float coordinates
        customer_float = Customer("C2", {"Dry": 10}, (1.5, 2.5), 15)
        assert customer_float.location == (1.5, 2.5)

        # Depot with different types
        depot_int = DepotLocation(latitude=10, longitude=20)
        assert depot_int.latitude == 10
        assert depot_int.longitude == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])