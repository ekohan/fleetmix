import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from fleetmix.utils.route_time import _legacy_estimation, _bhh_estimation, estimate_route_time


def test_legacy_estimation_zero_customers():
    assert _legacy_estimation(0, service_time=30) == pytest.approx(1.0)


def test_legacy_estimation_multiple_customers():
    # For 2 customers and 30 minutes each: 1 + 2*(30/60) = 1 + 1 = 2 hours
    assert _legacy_estimation(2, service_time=30) == pytest.approx(2.0)


def make_customers_df(coords):
    # coords: list of (lat, lon)
    df = pd.DataFrame({'Latitude':[c[0] for c in coords], 'Longitude':[c[1] for c in coords]})
    return df


def test_bhh_estimation_single_customer():
    # For single customer, should return service_time/60
    df = make_customers_df([(0,0)])
    est = _bhh_estimation(df, depot={'latitude':0, 'longitude':0}, service_time=30, avg_speed=60)
    assert est == pytest.approx(0.5)


def test_bhh_estimation_two_customers():
    # Two customers at unit distance 1 degree (~111 km), speed=111 km/h => travel ~1h roundtrip + service
    df = make_customers_df([(0,1),(0,-1)])
    est = _bhh_estimation(df, depot={'latitude':0,'longitude':0}, service_time=0, avg_speed=111)
    # Internally, BHH intra-cluster time = 0.765 * sqrt(n) * sqrt(pi * radius^2) / avg_speed
    # Here radius=~111 km cancels with avg_speed=111, so expected = 0.765 * sqrt(2) * sqrt(pi)
    expected = 0.765 * np.sqrt(2) * np.sqrt(np.pi)
    # Looser tolerance (0.3%) for numeric approximations
    assert est == pytest.approx(expected, rel=3e-3)


class TestRouteTime:
    """Test route time estimation functions"""
    
    def test_tsp_prune_warning(self):
        """Test TSP pruning logs warning when BHH estimate exceeds max route time"""
        # Create sample customer data
        cluster_customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [4.5, 4.6, 4.7],
            'Longitude': [-74.0, -74.1, -74.2]
        })
        
        depot = {'latitude': 4.65, 'longitude': -74.15}
        
        # Mock the BHH estimation to return a value that exceeds max_route_time
        with patch('fleetmix.utils.route_time._bhh_estimation') as mock_bhh:
            mock_bhh.return_value = 15.0  # BHH returns 15 hours
            
            with patch('fleetmix.utils.route_time.logger') as mock_logger:
                # Call with prune_tsp=True and max_route_time=10
                time, sequence = estimate_route_time(
                    cluster_customers,
                    depot,
                    service_time=15,
                    avg_speed=30,
                    method='TSP',
                    max_route_time=10,
                    prune_tsp=True
                )
                
                # Check that warning was logged
                mock_logger.warning.assert_any_call("Prune TSP: True, Max Route Time: 10")
                # Check that the skipped message was logged
                assert any("Cluster skipped TSP computation" in str(call) 
                          for call in mock_logger.warning.call_args_list)
                
                # Check return values when pruned
                assert time > 10  # Should return slightly over max_route_time
                assert sequence == []  # Empty sequence when pruned 