"""Test vehicle-specific goods capability."""

import pytest
import yaml
from pathlib import Path
from fleetmix.config.parameters import Parameters
from fleetmix.core_types import VehicleSpec
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations


class TestVehicleAllowedGoods:
    """Test suite for vehicle-specific goods functionality."""

    def test_vehicle_spec_with_allowed_goods(self):
        """Test VehicleSpec with allowed_goods field."""
        spec = VehicleSpec(
            capacity=3000,
            fixed_cost=100,
            allowed_goods=["Dry", "Chilled"]
        )
        assert spec.allowed_goods == ["Dry", "Chilled"]
        
        # Test to_dict includes allowed_goods
        spec_dict = spec.to_dict()
        assert "allowed_goods" in spec_dict
        assert spec_dict["allowed_goods"] == ["Dry", "Chilled"]
    
    def test_vehicle_spec_without_allowed_goods(self):
        """Test VehicleSpec without allowed_goods field (backward compatibility)."""
        spec = VehicleSpec(
            capacity=3000,
            fixed_cost=100
        )
        assert spec.allowed_goods is None
        
        # Test to_dict doesn't include allowed_goods when None
        spec_dict = spec.to_dict()
        assert "allowed_goods" not in spec_dict
    
    def test_parameters_validation_allowed_goods_valid(self, tmp_path):
        """Test Parameters validation with valid allowed_goods."""
        config_content = """
vehicles:
  A:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
  B:
    capacity: 3300
    fixed_cost: 175
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Dry", "Chilled"]
  C:
    capacity: 4500
    fixed_cost: 225
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Frozen"]

goods:
  - Dry
  - Chilled
  - Frozen

depot:
  latitude: 40.7128
  longitude: -74.0060

clustering:
  method: minibatch_kmeans
  max_depth: 3
  route_time_estimation: TSP
  geo_weight: 0.7
  demand_weight: 0.3

demand_file: test_demand.csv
light_load_penalty: 50
light_load_threshold: 0.5
compartment_setup_cost: 10
format: csv
variable_cost_per_hour: 50
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        # Should load without errors
        params = Parameters.from_yaml(config_file)
        
        assert params.vehicles["A"].allowed_goods is None
        assert params.vehicles["B"].allowed_goods == ["Dry", "Chilled"]
        assert params.vehicles["C"].allowed_goods == ["Frozen"]
    
    def test_parameters_validation_invalid_goods(self, tmp_path):
        """Test Parameters validation with goods not in global list."""
        config_content = """
vehicles:
  A:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Dry", "InvalidGood"]

goods:
  - Dry
  - Chilled
  - Frozen

depot:
  latitude: 40.7128
  longitude: -74.0060

clustering:
  method: minibatch_kmeans
  max_depth: 3
  route_time_estimation: TSP
  geo_weight: 0.7
  demand_weight: 0.3

demand_file: test_demand.csv
light_load_penalty: 50
light_load_threshold: 0.5
compartment_setup_cost: 10
format: csv
variable_cost_per_hour: 50
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(ValueError, match="allowed_goods contains goods not in global list"):
            Parameters.from_yaml(config_file)
    
    def test_parameters_validation_empty_allowed_goods(self, tmp_path):
        """Test Parameters validation with empty allowed_goods list."""
        config_content = """
vehicles:
  A:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: []

goods:
  - Dry
  - Chilled
  - Frozen

depot:
  latitude: 40.7128
  longitude: -74.0060

clustering:
  method: minibatch_kmeans
  max_depth: 3
  route_time_estimation: TSP
  geo_weight: 0.7
  demand_weight: 0.3

demand_file: test_demand.csv
light_load_penalty: 50
light_load_threshold: 0.5
compartment_setup_cost: 10
format: csv
variable_cost_per_hour: 50
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(ValueError, match="allowed_goods cannot be empty"):
            Parameters.from_yaml(config_file)
    
    def test_parameters_validation_duplicate_goods(self, tmp_path):
        """Test Parameters validation with duplicate goods in allowed_goods."""
        config_content = """
vehicles:
  A:
    capacity: 2700
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Dry", "Dry", "Chilled"]

goods:
  - Dry
  - Chilled
  - Frozen

depot:
  latitude: 40.7128
  longitude: -74.0060

clustering:
  method: minibatch_kmeans
  max_depth: 3
  route_time_estimation: TSP
  geo_weight: 0.7
  demand_weight: 0.3

demand_file: test_demand.csv
light_load_penalty: 50
light_load_threshold: 0.5
compartment_setup_cost: 10
format: csv
variable_cost_per_hour: 50
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(ValueError, match="allowed_goods contains duplicates"):
            Parameters.from_yaml(config_file)
    
    def test_generate_vehicle_configurations_with_allowed_goods(self):
        """Test vehicle configuration generation respects allowed_goods."""
        vehicles = {
            "A": VehicleSpec(
                capacity=2700,
                fixed_cost=100,
                allowed_goods=None  # Can carry all goods
            ),
            "B": VehicleSpec(
                capacity=3300,
                fixed_cost=175,
                allowed_goods=["Dry", "Chilled"]  # Cannot carry Frozen
            ),
            "C": VehicleSpec(
                capacity=4500,
                fixed_cost=225,
                allowed_goods=["Frozen"]  # Can only carry Frozen
            )
        }
        
        goods = ["Dry", "Chilled", "Frozen"]
        configs = generate_vehicle_configurations(vehicles, goods)
        
        # Vehicle A should have 2^3 - 1 = 7 configurations (all combinations except empty)
        a_configs = [c for c in configs if c.vehicle_type == "A"]
        assert len(a_configs) == 7
        
        # Vehicle B should have 2^2 - 1 = 3 configurations (Dry, Chilled, Dry+Chilled)
        b_configs = [c for c in configs if c.vehicle_type == "B"]
        assert len(b_configs) == 3
        # Verify B configs never have Frozen
        for config in b_configs:
            assert config.compartments["Frozen"] is False
            assert config.compartments["Dry"] or config.compartments["Chilled"]
        
        # Vehicle C should have 2^1 - 1 = 1 configuration (Frozen only)
        c_configs = [c for c in configs if c.vehicle_type == "C"]
        assert len(c_configs) == 1
        assert c_configs[0].compartments == {"Dry": False, "Chilled": False, "Frozen": True}
    
    def test_generate_vehicle_configurations_compartments_initialization(self):
        """Test that all goods are initialized in compartments dict."""
        vehicles = {
            "A": VehicleSpec(
                capacity=2700,
                fixed_cost=100,
                allowed_goods=["Dry"]  # Only allowed Dry
            )
        }
        
        goods = ["Dry", "Chilled", "Frozen"]
        configs = generate_vehicle_configurations(vehicles, goods)
        
        # Should have 1 configuration
        assert len(configs) == 1
        config = configs[0]
        
        # All goods should be in compartments dict
        assert set(config.compartments.keys()) == set(goods)
        # Only Dry should be True
        assert config.compartments["Dry"] is True
        assert config.compartments["Chilled"] is False
        assert config.compartments["Frozen"] is False
    
    def test_parameters_to_yaml_preserves_allowed_goods(self, tmp_path):
        """Test that to_yaml preserves allowed_goods."""
        # Create parameters with allowed_goods
        vehicles = {
            "A": VehicleSpec(
                capacity=2700,
                fixed_cost=100,
                allowed_goods=["Dry", "Chilled"]
            ),
            "B": VehicleSpec(
                capacity=3300,
                fixed_cost=175
            )
        }
        
        params = Parameters(
            vehicles=vehicles,
            variable_cost_per_hour=50,
            depot={"latitude": 40.7128, "longitude": -74.0060},
            goods=["Dry", "Chilled", "Frozen"],
            clustering={
                "method": "minibatch_kmeans",
                "max_depth": 3,
                "route_time_estimation": "TSP",
                "geo_weight": 0.7,
                "demand_weight": 0.3
            },
            demand_file="test_demand.csv",
            light_load_penalty=50,
            light_load_threshold=0.5,
            compartment_setup_cost=10,
            format="csv"
        )
        
        # Save to YAML
        output_file = tmp_path / "output_config.yaml"
        params.to_yaml(output_file)
        
        # Load and verify
        with open(output_file) as f:
            loaded_data = yaml.safe_load(f)
        
        # Debug: print the loaded data
        print(f"Vehicle B data: {loaded_data['vehicles']['B']}")
        
        assert loaded_data["vehicles"]["A"]["allowed_goods"] == ["Dry", "Chilled"]
        assert "allowed_goods" not in loaded_data["vehicles"]["B"] 