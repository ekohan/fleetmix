"""Test the route_time module comprehensively."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    build_distance_duration_matrices,
    estimate_route_time,
    _matrix_cache
)


def test_calculate_total_service_time_hours():
    """Test service time calculation."""
    # Normal case
    assert calculate_total_service_time_hours(5, 10) == 50/60  # 5 customers * 10 min / 60
    
    # Zero customers
    assert calculate_total_service_time_hours(0, 10) == 0.0
    
    # Zero service time
    assert calculate_total_service_time_hours(5, 0) == 0.0
    
    # Negative customers (should return 0)
    assert calculate_total_service_time_hours(-5, 10) == 0.0
    
    # Negative service time (should return 0)
    assert calculate_total_service_time_hours(5, -10) == 0.0


def test_build_distance_duration_matrices():
    """Test building distance and duration matrices."""
    # Clear cache first
    _matrix_cache['distance_matrix'] = None
    _matrix_cache['duration_matrix'] = None
    _matrix_cache['customer_id_to_idx'] = None
    
    # Create test data
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2'],
        'Latitude': [0.1, 0.2],
        'Longitude': [0.1, 0.2]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    avg_speed = 30  # km/h
    
    # Build matrices
    build_distance_duration_matrices(customers_df, depot, avg_speed)
    
    # Check cache is populated
    assert _matrix_cache['distance_matrix'] is not None
    assert _matrix_cache['duration_matrix'] is not None
    assert _matrix_cache['customer_id_to_idx'] is not None
    
    # Check dimensions (3x3 for depot + 2 customers)
    assert _matrix_cache['distance_matrix'].shape == (3, 3)
    assert _matrix_cache['duration_matrix'].shape == (3, 3)
    
    # Check customer mapping
    assert _matrix_cache['customer_id_to_idx'] == {'C1': 1, 'C2': 2}
    
    # Check diagonal is zero
    assert np.diag(_matrix_cache['distance_matrix']).sum() == 0
    assert np.diag(_matrix_cache['duration_matrix']).sum() == 0


def test_build_distance_duration_matrices_empty():
    """Test building matrices with empty customer data."""
    # Clear cache
    _matrix_cache['distance_matrix'] = None
    
    customers_df = pd.DataFrame()
    depot = {'latitude': 0.0, 'longitude': 0.0}
    
    # Should log warning and return
    build_distance_duration_matrices(customers_df, depot, 30)
    
    # Cache should remain None
    assert _matrix_cache['distance_matrix'] is None


def test_build_distance_duration_matrices_missing_columns():
    """Test building matrices with missing coordinate columns."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1'],
        # Missing Latitude and Longitude
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Missing coordinate columns"):
        build_distance_duration_matrices(customers_df, depot, 30)


def test_estimate_route_time_legacy():
    """Test legacy route time estimation."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2'],
        'Latitude': [0.1, 0.2],
        'Longitude': [0.1, 0.2]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 10  # minutes
    
    time, sequence = estimate_route_time(
        customers_df, depot, service_time, 30, method='Legacy'
    )
    
    # Legacy always returns 1 hour + service time
    expected = 1 + (2 * 10 / 60)  # 1 + 20/60 hours
    assert time == expected
    assert sequence == []  # Empty sequence for non-TSP methods


def test_estimate_route_time_bhh():
    """Test BHH route time estimation."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2', 'C3'],
        'Latitude': [0.1, 0.15, 0.2],
        'Longitude': [0.1, 0.15, 0.2]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 10  # minutes
    avg_speed = 30  # km/h
    
    time, sequence = estimate_route_time(
        customers_df, depot, service_time, avg_speed, method='BHH'
    )
    
    # BHH should return positive time
    assert time > 0
    assert sequence == []  # Empty sequence for non-TSP methods


def test_estimate_route_time_bhh_single_customer():
    """Test BHH estimation with single customer."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1'],
        'Latitude': [0.1],
        'Longitude': [0.1]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 10  # minutes
    
    time, sequence = estimate_route_time(
        customers_df, depot, service_time, 30, method='BHH'
    )
    
    # For single customer, should just be service time
    assert time == 10 / 60  # 10 minutes in hours


def test_estimate_route_time_invalid_method():
    """Test with invalid estimation method."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1'],
        'Latitude': [0.1],
        'Longitude': [0.1]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    
    with pytest.raises(ValueError, match="Unknown route time estimation method"):
        estimate_route_time(customers_df, depot, 10, 30, method='INVALID')


def test_legacy_estimation_through_public_api():
    """Test the legacy estimation behavior through public API."""
    # 5 customers, 10 min service time each
    customers_df = pd.DataFrame({
        'Customer_ID': [f'C{i}' for i in range(5)],
        'Latitude': [0.1 * i for i in range(5)],
        'Longitude': [0.1 * i for i in range(5)]
    })
    depot = {'latitude': 0.0, 'longitude': 0.0}
    
    time, sequence = estimate_route_time(customers_df, depot, 10, 30, method='Legacy')
    assert time == 1 + (5 * 10 / 60)  # 1 + 50/60 hours
    assert sequence == []


def test_bhh_estimation_through_public_api():
    """Test the BHH estimation behavior through public API."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2'],
        'Latitude': [0.1, 0.2],
        'Longitude': [0.1, 0.2]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 10  # minutes
    avg_speed = 30  # km/h
    
    time, sequence = estimate_route_time(customers_df, depot, service_time, avg_speed, method='BHH')
    
    # Should return positive time
    assert time > 0
    # Should include service time component
    assert time >= (2 * 10 / 60)  # At least the service time


def test_tsp_estimation_zero_customers():
    """Test TSP estimation with zero customers."""
    customers_df = pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude'])
    depot = {'latitude': 0.0, 'longitude': 0.0}
    
    time, sequence = estimate_route_time(customers_df, depot, 10, 30, method='TSP')
    
    assert time == 0.0
    assert sequence == []


def test_tsp_estimation_single_customer():
    """Test TSP estimation with single customer."""
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1'],
        'Latitude': [0.1],
        'Longitude': [0.1]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 10  # minutes
    avg_speed = 30  # km/h
    
    time, sequence = estimate_route_time(customers_df, depot, service_time, avg_speed, method='TSP')
    
    # Should return travel time + service time
    assert time > 0
    assert time >= 10 / 60  # At least service time
    assert sequence == ["Depot", "C1", "Depot"]


@patch('fleetmix.utils.route_time.Model')
def test_tsp_estimation_infeasible(mock_model):
    """Test TSP estimation when solution is infeasible."""
    # Mock PyVRP to return infeasible solution
    mock_result = MagicMock()
    mock_result.best.is_feasible.return_value = False
    mock_model.from_data.return_value.solve.return_value = mock_result
    
    customers_df = pd.DataFrame({
        'Customer_ID': ['C1', 'C2'],
        'Latitude': [0.1, 0.2],
        'Longitude': [0.1, 0.2]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    max_route_time = 8  # hours
    
    time, sequence = estimate_route_time(
        customers_df, depot, 10, 30, method='TSP', max_route_time=max_route_time
    )
    
    # Should return slightly over max_route_time
    assert time > max_route_time
    assert sequence == []


def test_estimate_route_time_tsp_with_pruning():
    """Test TSP estimation with pruning enabled."""
    customers_df = pd.DataFrame({
        'Customer_ID': [f'C{i}' for i in range(20)],  # Many customers
        'Latitude': [0.1 * i for i in range(20)],
        'Longitude': [0.1 * i for i in range(20)]
    })
    
    depot = {'latitude': 0.0, 'longitude': 0.0}
    service_time = 30  # Long service time
    avg_speed = 20  # Slow speed
    max_route_time = 2  # Very short max time
    
    # With pruning, should skip TSP computation
    time, sequence = estimate_route_time(
        customers_df, depot, service_time, avg_speed,
        method='TSP', max_route_time=max_route_time, prune_tsp=True
    )
    
    # Should return slightly over max time
    assert time > max_route_time
    assert sequence == [] 