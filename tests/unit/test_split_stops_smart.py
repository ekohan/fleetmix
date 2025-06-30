"""Test smart split-stop functionality with vehicle-specific goods constraints."""

import pandas as pd
import pytest

from fleetmix.core_types import VehicleConfiguration
from fleetmix.preprocess.demand import (
    explode_customer_smart,
    get_feasible_goods_combinations,
    maybe_explode,
)


class TestSmartSplitStops:
    """Test suite for smart split-stop functionality."""

    def test_get_feasible_goods_combinations(self):
        """Test getting feasible goods combinations based on vehicle configurations."""
        # Create vehicle configurations with specific allowed goods
        configs = [
            VehicleConfiguration(
                config_id=1,
                vehicle_type="DryOnly",
                capacity=2700,
                fixed_cost=100,
                compartments={"Dry": True, "Chilled": False, "Frozen": False}
            ),
            VehicleConfiguration(
                config_id=2,
                vehicle_type="ChilledOnly",
                capacity=3300,
                fixed_cost=175,
                compartments={"Dry": False, "Chilled": True, "Frozen": False}
            ),
            VehicleConfiguration(
                config_id=3,
                vehicle_type="FrozenOnly",
                capacity=4500,
                fixed_cost=225,
                compartments={"Dry": False, "Chilled": False, "Frozen": True}
            ),
        ]
        
        # Test 1: Customer needs Dry and Chilled
        goods_with_demand = ["Dry", "Chilled"]
        combinations = get_feasible_goods_combinations(goods_with_demand, configs)
        
        # Should only get individual goods, not the combination
        assert sorted(combinations) == [("Chilled",), ("Dry",)]
        
        # Test 2: Customer needs all three goods
        goods_with_demand = ["Dry", "Chilled", "Frozen"]
        combinations = get_feasible_goods_combinations(goods_with_demand, configs)
        
        # Should only get individual goods
        assert sorted(combinations) == [("Chilled",), ("Dry",), ("Frozen",)]
        
        # Test 3: Add a multi-compartment vehicle
        configs.append(
            VehicleConfiguration(
                config_id=4,
                vehicle_type="MultiTemp",
                capacity=5000,
                fixed_cost=300,
                compartments={"Dry": False, "Chilled": True, "Frozen": True}
            )
        )
        
        goods_with_demand = ["Chilled", "Frozen"]
        combinations = get_feasible_goods_combinations(goods_with_demand, configs)
        
        # Now should include the combination
        assert sorted(combinations) == [("Chilled",), ("Chilled", "Frozen"), ("Frozen",)]

    def test_explode_customer_smart(self):
        """Test smart customer explosion with vehicle constraints."""
        # Create vehicle configurations
        configs = [
            VehicleConfiguration(
                config_id=1,
                vehicle_type="DryOnly",
                capacity=2700,
                fixed_cost=100,
                compartments={"Dry": True, "Chilled": False, "Frozen": False}
            ),
            VehicleConfiguration(
                config_id=2,
                vehicle_type="ChilledFrozen",
                capacity=3300,
                fixed_cost=175,
                compartments={"Dry": False, "Chilled": True, "Frozen": True}
            ),
        ]
        
        # Customer with demands that no single vehicle can serve
        customer_id = "C001"
        demands = {"Dry": 100, "Chilled": 50, "Frozen": 30}
        location = (40.7128, -74.0060)
        
        pseudo_customers = explode_customer_smart(
            customer_id, demands, location, configs
        )
        
        # Should create 4 pseudo-customers:
        # - Dry (served by DryOnly)
        # - Chilled (served by ChilledFrozen)
        # - Frozen (served by ChilledFrozen)
        # - Chilled-Frozen (served by ChilledFrozen)
        assert len(pseudo_customers) == 4
        
        # Check pseudo-customer IDs
        pseudo_ids = [pc.customer_id for pc in pseudo_customers]
        assert "C001::Dry" in pseudo_ids
        assert "C001::Chilled" in pseudo_ids
        assert "C001::Frozen" in pseudo_ids
        assert "C001::Chilled-Frozen" in pseudo_ids
        
        # Verify demands are correctly split
        for pc in pseudo_customers:
            if pc.customer_id == "C001::Dry":
                assert pc.demands == {"Dry": 100, "Chilled": 0, "Frozen": 0}
            elif pc.customer_id == "C001::Chilled":
                assert pc.demands == {"Dry": 0, "Chilled": 50, "Frozen": 0}
            elif pc.customer_id == "C001::Frozen":
                assert pc.demands == {"Dry": 0, "Chilled": 0, "Frozen": 30}
            elif pc.customer_id == "C001::Chilled-Frozen":
                assert pc.demands == {"Dry": 0, "Chilled": 50, "Frozen": 30}

    def test_maybe_explode_with_configurations(self):
        """Test maybe_explode with vehicle configurations."""
        # Create test data
        customers_df = pd.DataFrame({
            "Customer_ID": ["C001", "C002"],
            "Latitude": [40.7128, 40.7580],
            "Longitude": [-74.0060, -73.9855],
            "Dry_Demand": [100, 0],
            "Chilled_Demand": [50, 200],
            "Frozen_Demand": [0, 150],
            "Service_Time": [25, 25]
        })
        
        # Create vehicle configurations
        configs = [
            VehicleConfiguration(
                config_id=1,
                vehicle_type="DryOnly",
                capacity=2700,
                fixed_cost=100,
                compartments={"Dry": True, "Chilled": False, "Frozen": False}
            ),
            VehicleConfiguration(
                config_id=2,
                vehicle_type="ChilledOnly",
                capacity=3300,
                fixed_cost=175,
                compartments={"Dry": False, "Chilled": True, "Frozen": False}
            ),
            VehicleConfiguration(
                config_id=3,
                vehicle_type="FrozenOnly",
                capacity=4500,
                fixed_cost=225,
                compartments={"Dry": False, "Chilled": False, "Frozen": True}
            ),
        ]
        
        # Test with split stops enabled and configurations
        result = maybe_explode(customers_df, allow_split_stops=True, configurations=configs)
        
        # C001 needs Dry and Chilled -> 2 pseudo-customers
        # C002 needs Chilled and Frozen -> 2 pseudo-customers
        # Total: 4 pseudo-customers
        assert len(result) == 4
        
        # Check that infeasible combinations were not created
        customer_ids = result["Customer_ID"].tolist()
        assert "C001::Dry-Chilled" not in customer_ids  # No vehicle can serve this
        assert "C002::Chilled-Frozen" not in customer_ids  # No vehicle can serve this
        
        # Verify the correct pseudo-customers were created
        assert "C001::Dry" in customer_ids
        assert "C001::Chilled" in customer_ids
        assert "C002::Chilled" in customer_ids
        assert "C002::Frozen" in customer_ids 