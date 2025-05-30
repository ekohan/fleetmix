"""
Comprehensive tests for the route time estimation module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    build_distance_duration_matrices,
    estimate_route_time,
    _legacy_estimation,
    _bhh_estimation,
    _pyvrp_tsp_estimation,
    _matrix_cache
)


class TestServiceTimeCalculation:
    """Test suite for service time calculations"""
    
    def test_calculate_service_time_normal(self):
        """Test normal service time calculation"""
        result = calculate_total_service_time_hours(3, 20)  # 3 customers, 20 min each
        expected = 3 * 20 / 60  # 1 hour
        assert result == expected
    
    def test_calculate_service_time_zero_customers(self):
        """Test service time with zero customers"""
        result = calculate_total_service_time_hours(0, 30)
        assert result == 0.0
    
    def test_calculate_service_time_negative_customers(self):
        """Test service time with negative customers (should warn and return 0)"""
        result = calculate_total_service_time_hours(-1, 30)
        assert result == 0.0
    
    def test_calculate_service_time_negative_time(self):
        """Test service time with negative time (should warn and return 0)"""
        result = calculate_total_service_time_hours(2, -10)
        assert result == 0.0


class TestDistanceMatrixBuilding:
    """Test suite for distance matrix building"""
    
    def setup_method(self):
        """Reset matrix cache before each test"""
        _matrix_cache['distance_matrix'] = None
        _matrix_cache['duration_matrix'] = None
        _matrix_cache['customer_id_to_idx'] = None
        _matrix_cache['depot_idx'] = 0
    
    def test_build_matrices_normal(self):
        """Test building distance and duration matrices"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        build_distance_duration_matrices(customers_df, depot, avg_speed=50)
        
        # Check cache was populated
        assert _matrix_cache['distance_matrix'] is not None
        assert _matrix_cache['duration_matrix'] is not None
        assert _matrix_cache['customer_id_to_idx'] is not None
        
        # Check dimensions (depot + 2 customers = 3x3)
        assert _matrix_cache['distance_matrix'].shape == (3, 3)
        assert _matrix_cache['duration_matrix'].shape == (3, 3)
        
        # Check customer ID mapping
        assert _matrix_cache['customer_id_to_idx']['C1'] == 1
        assert _matrix_cache['customer_id_to_idx']['C2'] == 2
    
    def test_build_matrices_empty_dataframe(self):
        """Test building matrices with empty customer DataFrame"""
        customers_df = pd.DataFrame()
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        build_distance_duration_matrices(customers_df, depot, avg_speed=50)
        
        # Cache should remain empty
        assert _matrix_cache['distance_matrix'] is None
    
    def test_build_matrices_missing_coordinates(self):
        """Test error when coordinate columns are missing"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128]
            # Missing Longitude
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        with pytest.raises(ValueError, match="Missing coordinate columns"):
            build_distance_duration_matrices(customers_df, depot, avg_speed=50)
    
    def test_build_matrices_duplicate_customer_ids(self):
        """Test warning with duplicate customer IDs"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C1'],  # Duplicate IDs
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        # Should build matrices but warn about duplicates
        build_distance_duration_matrices(customers_df, depot, avg_speed=50)
        assert _matrix_cache['distance_matrix'] is not None


class TestLegacyEstimation:
    """Test suite for legacy estimation method"""
    
    def test_legacy_zero_customers(self):
        """Test legacy estimation with zero customers"""
        result = _legacy_estimation(0, 30)
        assert result == 1.0  # Only travel time component
    
    def test_legacy_multiple_customers(self):
        """Test legacy estimation with multiple customers"""
        result = _legacy_estimation(4, 15)  # 4 customers, 15 min each
        expected = 1.0 + (4 * 15 / 60)  # 1 + 1 = 2 hours
        assert result == expected


class TestBHHEstimation:
    """Test suite for BHH estimation method"""
    
    def test_bhh_zero_customers(self):
        """Test BHH estimation with zero customers"""
        customers_df = pd.DataFrame()
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        result = _bhh_estimation(customers_df, depot, 30, 50)
        assert result == 0.0
    
    def test_bhh_single_customer(self):
        """Test BHH estimation with single customer"""
        customers_df = pd.DataFrame({
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        result = _bhh_estimation(customers_df, depot, 30, 50)
        expected = 30 / 60  # Just service time
        assert result == pytest.approx(expected)
    
    def test_bhh_multiple_customers(self):
        """Test BHH estimation with multiple customers"""
        customers_df = pd.DataFrame({
            'Latitude': [40.7128, 40.7589, 40.7505],
            'Longitude': [-74.0060, -73.9851, -73.9934]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        result = _bhh_estimation(customers_df, depot, 20, 40)
        # Should be positive and reasonable
        assert result > 0
        assert result < 24  # Less than a day


class TestRouteTimeEstimation:
    """Test suite for main estimate_route_time function"""
    
    def setup_method(self):
        """Reset matrix cache before each test"""
        _matrix_cache['distance_matrix'] = None
        _matrix_cache['duration_matrix'] = None
        _matrix_cache['customer_id_to_idx'] = None
        _matrix_cache['depot_idx'] = 0
    
    def test_estimate_route_time_legacy(self):
        """Test route time estimation using Legacy method"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        time, sequence = estimate_route_time(
            customers_df, depot, 20, 50, method='Legacy'
        )
        
        # Should match legacy calculation
        expected = 1.0 + (2 * 20 / 60)  # 1 + 0.67 ≈ 1.67 hours
        assert time == pytest.approx(expected)
        assert sequence == []  # Legacy doesn't return sequence
    
    def test_estimate_route_time_bhh(self):
        """Test route time estimation using BHH method"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        time, sequence = estimate_route_time(
            customers_df, depot, 20, 50, method='BHH'
        )
        
        assert time > 0
        assert sequence == []  # BHH doesn't return sequence
    
    def test_estimate_route_time_invalid_method(self):
        """Test error with invalid estimation method"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        with pytest.raises(ValueError, match="Unknown route time estimation method"):
            estimate_route_time(customers_df, depot, 20, 50, method='InvalidMethod')


class TestTSPEstimation:
    """Test suite for TSP estimation method"""
    
    def setup_method(self):
        """Reset matrix cache before each test"""
        _matrix_cache['distance_matrix'] = None
        _matrix_cache['duration_matrix'] = None
        _matrix_cache['customer_id_to_idx'] = None
        _matrix_cache['depot_idx'] = 0
    
    def test_tsp_zero_customers(self):
        """Test TSP estimation with zero customers"""
        customers_df = pd.DataFrame()
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        time, sequence = _pyvrp_tsp_estimation(customers_df, depot, 20, 50)
        assert time == 0.0
        assert sequence == []
    
    def test_tsp_single_customer(self):
        """Test TSP estimation with single customer"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        time, sequence = _pyvrp_tsp_estimation(customers_df, depot, 20, 50)
        
        assert time > 0
        assert len(sequence) == 3  # Depot -> Customer -> Depot
        assert sequence[0] == "Depot"
        assert sequence[1] == "C1"
        assert sequence[2] == "Depot"
    
    @patch('fleetmix.utils.route_time.Model')
    def test_tsp_with_cache(self, mock_model):
        """Test TSP estimation using cached matrices"""
        # Setup cache
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        # Build cache first
        build_distance_duration_matrices(customers_df, depot, avg_speed=50)
        
        # Mock PyVRP components
        mock_result = MagicMock()
        mock_result.best.is_feasible.return_value = True
        mock_result.best.duration.return_value = 3600  # 1 hour in seconds
        mock_route = MagicMock()
        mock_route.visits.return_value = [1, 2]  # Customer indices
        mock_result.best.routes.return_value = [mock_route]
        
        mock_model_instance = MagicMock()
        mock_model_instance.solve.return_value = mock_result
        mock_model.from_data.return_value = mock_model_instance
        
        time, sequence = _pyvrp_tsp_estimation(customers_df, depot, 20, 50)
        
        assert time == 1.0  # 3600 seconds = 1 hour
        assert len(sequence) == 4  # Depot -> C1 -> C2 -> Depot
    
    @patch('fleetmix.utils.route_time.Model')
    def test_tsp_infeasible_solution(self, mock_model):
        """Test TSP estimation when solution is infeasible"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        # Mock infeasible result
        mock_result = MagicMock()
        mock_result.best.is_feasible.return_value = False
        
        mock_model_instance = MagicMock()
        mock_model_instance.solve.return_value = mock_result
        mock_model.from_data.return_value = mock_model_instance
        
        time, sequence = _pyvrp_tsp_estimation(customers_df, depot, 20, 50, max_route_time=8)
        
        # For single customer, the function falls back to direct calculation
        # so it returns a valid time and sequence even when PyVRP says infeasible
        assert time > 0
        # Single customer case returns sequence regardless of feasibility
        assert len(sequence) == 3  # Depot -> Customer -> Depot
    
    def test_tsp_prune_with_bhh(self):
        """Test TSP pruning based on BHH estimate"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [40.7128, 40.7589],
            'Longitude': [-74.0060, -73.9851]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        # Use a very small max_route_time to trigger pruning
        time, sequence = estimate_route_time(
            customers_df, depot, 20, 50, 
            method='TSP', max_route_time=0.1, prune_tsp=True
        )
        
        # Should be pruned and return slightly over max_route_time
        assert time > 0.1
        assert sequence == []
    
    def test_tsp_on_the_fly_matrices(self):
        """Test TSP estimation without cached matrices"""
        customers_df = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        })
        depot = {'latitude': 40.7282, 'longitude': -73.9942}
        
        # Ensure cache is empty
        assert _matrix_cache['distance_matrix'] is None
        
        with patch('fleetmix.utils.route_time.Model') as mock_model:
            # Mock PyVRP components
            mock_result = MagicMock()
            mock_result.best.is_feasible.return_value = True
            mock_result.best.duration.return_value = 1800  # 30 minutes
            mock_route = MagicMock()
            mock_route.visits.return_value = [1]  # One customer
            mock_result.best.routes.return_value = [mock_route]
            
            mock_model_instance = MagicMock()
            mock_model_instance.solve.return_value = mock_result
            mock_model.from_data.return_value = mock_model_instance
            
            time, sequence = _pyvrp_tsp_estimation(customers_df, depot, 20, 50)
            
            # Should be positive time and proper sequence
            assert time > 0
            assert len(sequence) == 3  # Depot -> Customer -> Depot 