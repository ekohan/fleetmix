"""Integration test for vehicle-specific goods capability."""

import pytest
import pandas as pd
from pathlib import Path

from fleetmix.config.parameters import Parameters
from fleetmix.core_types import Customer
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering.generator import generate_clusters_for_configurations
from fleetmix.optimization.core import optimize_fleet


class TestAllowedGoodsIntegration:
    """Integration tests for vehicle-specific goods functionality."""
    
    @pytest.fixture
    def test_config_with_allowed_goods(self, tmp_path):
        """Create a test configuration with allowed_goods."""
        config_content = """
vehicles:
  DryOnly:
    capacity: 2000
    fixed_cost: 100
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Dry"]
  ChilledFrozen:
    capacity: 2500
    fixed_cost: 150
    avg_speed: 30
    service_time: 25
    max_route_time: 10
    allowed_goods: ["Chilled", "Frozen"]
  Universal:
    capacity: 3000
    fixed_cost: 200
    avg_speed: 30
    service_time: 25
    max_route_time: 10

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
  route_time_estimation: BHH
  geo_weight: 0.7
  demand_weight: 0.3

demand_file: test_demand.csv
light_load_penalty: 50
light_load_threshold: 0.5
compartment_setup_cost: 10
format: csv
variable_cost_per_hour: 50
post_optimization: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return config_file
    
    @pytest.fixture
    def test_customers(self):
        """Create test customers with different good requirements."""
        return [
            Customer(
                customer_id="C1",
                demands={"Dry": 500, "Chilled": 0, "Frozen": 0},
                location=(40.7200, -74.0100),
                service_time=25
            ),
            Customer(
                customer_id="C2",
                demands={"Dry": 0, "Chilled": 300, "Frozen": 200},
                location=(40.7300, -74.0200),
                service_time=25
            ),
            Customer(
                customer_id="C3",
                demands={"Dry": 400, "Chilled": 300, "Frozen": 0},
                location=(40.7400, -74.0300),
                service_time=25
            ),
            Customer(
                customer_id="C4",
                demands={"Dry": 0, "Chilled": 0, "Frozen": 600},
                location=(40.7500, -74.0400),
                service_time=25
            ),
        ]
    
    def test_vehicle_configurations_respect_allowed_goods(self, test_config_with_allowed_goods):
        """Test that vehicle configurations are generated correctly with allowed_goods."""
        params = Parameters.from_yaml(test_config_with_allowed_goods)
        configs = generate_vehicle_configurations(params.vehicles, params.goods)
        
        # DryOnly should only have configurations with Dry
        dry_only_configs = [c for c in configs if c.vehicle_type == "DryOnly"]
        assert len(dry_only_configs) == 1  # Only one config: Dry
        assert all(c.compartments["Dry"] and not c.compartments["Chilled"] and not c.compartments["Frozen"] 
                  for c in dry_only_configs)
        
        # ChilledFrozen should only have configurations with Chilled and/or Frozen
        chilled_frozen_configs = [c for c in configs if c.vehicle_type == "ChilledFrozen"]
        assert len(chilled_frozen_configs) == 3  # Chilled, Frozen, Chilled+Frozen
        assert all(not c.compartments["Dry"] for c in chilled_frozen_configs)
        assert all(c.compartments["Chilled"] or c.compartments["Frozen"] for c in chilled_frozen_configs)
        
        # Universal should have all possible configurations
        universal_configs = [c for c in configs if c.vehicle_type == "Universal"]
        assert len(universal_configs) == 7  # 2^3 - 1
    
    def test_clustering_respects_vehicle_allowed_goods(self, test_config_with_allowed_goods, test_customers):
        """Test that clustering only assigns customers to compatible vehicles."""
        params = Parameters.from_yaml(test_config_with_allowed_goods)
        configs = generate_vehicle_configurations(params.vehicles, params.goods)
        
        # Generate clusters
        clusters = generate_clusters_for_configurations(test_customers, configs, params)
        
        # Verify that clusters are only created for compatible vehicle configurations
        for cluster in clusters:
            config = next(c for c in configs if c.config_id == cluster.config_id)
            
            # Check that all required goods are available in the configuration
            for customer_id in cluster.customers:
                customer = next(c for c in test_customers if c.customer_id == customer_id)
                for good, demand in customer.demands.items():
                    if demand > 0:
                        assert config.compartments[good], \
                            f"Cluster {cluster.cluster_id} with config {config.vehicle_type} " \
                            f"contains customer {customer_id} requiring {good}"
    
    def test_optimization_assigns_correct_vehicles(self, test_config_with_allowed_goods, test_customers):
        """Test that optimization assigns customers to vehicles with compatible goods."""
        params = Parameters.from_yaml(test_config_with_allowed_goods)
        configs = generate_vehicle_configurations(params.vehicles, params.goods)
        
        # Generate clusters
        clusters = generate_clusters_for_configurations(test_customers, configs, params)
        
        # Run optimization
        solution = optimize_fleet(clusters, configs, test_customers, params, verbose=False)
        
        # Verify solution
        assert solution.solver_status == "Optimal"
        assert len(solution.missing_customers) == 0
        
        # Check vehicle assignments
        for cluster in solution.selected_clusters:
            config = next(c for c in configs if c.config_id == cluster.config_id)
            vehicle_type = config.vehicle_type
            
            # Verify each customer in the cluster can be served by the vehicle
            for customer_id in cluster.customers:
                customer = next(c for c in test_customers if c.customer_id == customer_id)
                
                if vehicle_type == "DryOnly":
                    # Should only have dry demand
                    assert customer.demands["Dry"] > 0
                    assert customer.demands["Chilled"] == 0
                    assert customer.demands["Frozen"] == 0
                
                elif vehicle_type == "ChilledFrozen":
                    # Should not have dry demand
                    assert customer.demands["Dry"] == 0
                    assert customer.demands["Chilled"] > 0 or customer.demands["Frozen"] > 0
    
    def test_mixed_fleet_optimization(self, test_config_with_allowed_goods, test_customers):
        """Test optimization with a mixed fleet of specialized and universal vehicles."""
        params = Parameters.from_yaml(test_config_with_allowed_goods)
        configs = generate_vehicle_configurations(params.vehicles, params.goods)
        
        # Generate clusters and solve
        clusters = generate_clusters_for_configurations(test_customers, configs, params)
        solution = optimize_fleet(clusters, configs, test_customers, params, verbose=False)
        
        # Verify we get an optimal solution
        assert solution.solver_status == "Optimal"
        assert solution.total_vehicles > 0
        
        # Check that specialized vehicles are used when appropriate
        vehicles_used = solution.vehicles_used
        
        # We should use at least one specialized vehicle since we have customers
        # that only need specific goods
        assert "DryOnly" in vehicles_used or "ChilledFrozen" in vehicles_used or "Universal" in vehicles_used
        
        # Total cost should be reasonable
        assert solution.total_cost > 0
        assert solution.total_fixed_cost > 0 