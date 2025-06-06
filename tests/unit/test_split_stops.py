"""
Unit tests for split-stop functionality.
"""

import pytest
import pandas as pd
from fleetmix.preprocess.demand import explode_customer, maybe_explode, is_pseudo_customer, get_origin_id, get_subset_from_id
from fleetmix.core_types import PseudoCustomer


class TestExplodeCustomer:
    """Test the explode_customer function."""
    
    def test_single_good_customer(self):
        """Test exploding a customer with only one good type."""
        demands = {"dry": 10.0, "chilled": 0.0, "frozen": 0.0}
        location = (40.7, -74.0)
        
        pseudo_customers = explode_customer("C001", demands, location)
        
        assert len(pseudo_customers) == 1
        assert pseudo_customers[0].customer_id == "C001::dry"
        assert pseudo_customers[0].origin_id == "C001"
        assert pseudo_customers[0].subset == ("dry",)
        assert pseudo_customers[0].demands == {"dry": 10.0, "chilled": 0.0, "frozen": 0.0}
        assert pseudo_customers[0].location == location
    
    def test_two_goods_customer(self):
        """Test exploding a customer with two good types."""
        demands = {"dry": 10.0, "chilled": 5.0, "frozen": 0.0}
        location = (40.7, -74.0)
        
        pseudo_customers = explode_customer("C001", demands, location)
        
        # Should generate 3 pseudo-customers: {dry}, {chilled}, {dry, chilled}
        assert len(pseudo_customers) == 3
        
        # Check all customer IDs are correct
        customer_ids = [pc.customer_id for pc in pseudo_customers]
        expected_ids = ["C001::dry", "C001::chilled", "C001::dry-chilled"]
        for expected_id in expected_ids:
            assert expected_id in customer_ids
        
        # Check the combined subset has all demands
        combined_pseudo = next(pc for pc in pseudo_customers if pc.customer_id == "C001::dry-chilled")
        assert combined_pseudo.demands == {"dry": 10.0, "chilled": 5.0, "frozen": 0.0}
        
        # Check individual subsets have correct demands
        dry_only = next(pc for pc in pseudo_customers if pc.customer_id == "C001::dry")
        assert dry_only.demands == {"dry": 10.0, "chilled": 0.0, "frozen": 0.0}
        
        chilled_only = next(pc for pc in pseudo_customers if pc.customer_id == "C001::chilled")
        assert chilled_only.demands == {"dry": 0.0, "chilled": 5.0, "frozen": 0.0}
    
    def test_three_goods_customer(self):
        """Test exploding a customer with all three good types."""
        demands = {"dry": 10.0, "chilled": 5.0, "frozen": 3.0}
        location = (40.7, -74.0)
        
        pseudo_customers = explode_customer("C001", demands, location)
        
        # Should generate 7 pseudo-customers (2^3 - 1 = 7 non-empty subsets)
        assert len(pseudo_customers) == 7
        
        # Check that all origin_ids are correct
        for pc in pseudo_customers:
            assert pc.origin_id == "C001"
            assert pc.location == location
    
    def test_zero_demand_customer_raises_assertion(self):
        """Test that exploding a customer with zero demands raises an assertion error."""
        demands = {"dry": 0.0, "chilled": 0.0, "frozen": 0.0}
        location = (40.7, -74.0)
        
        # Should raise an AssertionError for invalid zero demands
        with pytest.raises(AssertionError):
            explode_customer("C001", demands, location)


class TestMaybeExplode:
    """Test the maybe_explode function."""
    
    def test_split_stops_disabled(self):
        """Test that original data is returned when split-stops is disabled."""
        customers_df = pd.DataFrame([
            {"Customer_ID": "C001", "Latitude": 40.7, "Longitude": -74.0, 
             "Dry_Demand": 10, "Chilled_Demand": 5, "Frozen_Demand": 0},
            {"Customer_ID": "C002", "Latitude": 40.8, "Longitude": -74.1, 
             "Dry_Demand": 0, "Chilled_Demand": 0, "Frozen_Demand": 8}
        ])
        
        result = maybe_explode(customers_df, allow_split_stops=False)
        
        # Should return exact copy of original
        pd.testing.assert_frame_equal(result, customers_df)
    
    def test_split_stops_enabled(self):
        """Test that pseudo-customers are created when split-stops is enabled."""
        customers_df = pd.DataFrame([
            {"Customer_ID": "C001", "Latitude": 40.7, "Longitude": -74.0, 
             "Dry_Demand": 10, "Chilled_Demand": 5, "Frozen_Demand": 0},
            {"Customer_ID": "C002", "Latitude": 40.8, "Longitude": -74.1, 
             "Dry_Demand": 0, "Chilled_Demand": 0, "Frozen_Demand": 8}
        ])
        
        result = maybe_explode(customers_df, allow_split_stops=True)
        
        # Should have more rows than original (C001 -> 3 pseudo, C002 -> 1 pseudo)
        assert len(result) == 4
        
        # Check that all pseudo-customers have Origin_ID column
        assert "Origin_ID" in result.columns
        assert "Subset" in result.columns
        
        # Check C001 pseudo-customers
        c001_pseudos = result[result["Origin_ID"] == "C001"]
        assert len(c001_pseudos) == 3  # dry, chilled, dry-chilled
        
        # Check C002 pseudo-customers  
        c002_pseudos = result[result["Origin_ID"] == "C002"]
        assert len(c002_pseudos) == 1  # only frozen


class TestUtilityFunctions:
    """Test utility functions for split-stop handling."""
    
    def test_is_pseudo_customer(self):
        """Test pseudo-customer identification."""
        assert is_pseudo_customer("C001::dry")
        assert is_pseudo_customer("C001::dry-chilled")
        assert not is_pseudo_customer("C001")
        assert not is_pseudo_customer("regular_customer")
    
    def test_get_origin_id(self):
        """Test extracting origin ID."""
        assert get_origin_id("C001::dry") == "C001"
        assert get_origin_id("C001::dry-chilled") == "C001"
        assert get_origin_id("C001") == "C001"  # Regular customer
    
    def test_get_subset_from_id(self):
        """Test extracting subset from pseudo-customer ID."""
        assert get_subset_from_id("C001::dry") == ("dry",)
        assert get_subset_from_id("C001::dry-chilled") == ("dry", "chilled")
        assert get_subset_from_id("C001::dry-chilled-frozen") == ("dry", "chilled", "frozen")
        assert get_subset_from_id("C001") == tuple()  # Regular customer 