"""Test route_time methods (BHH, TSP, Legacy) with flexible value ranges."""

import pytest
import pandas as pd
import numpy as np

from fleetmix.utils.route_time import estimate_route_time


class TestRouteTimeMethods:
    """Test different route time estimation methods with flexible value ranges."""
    
    @pytest.fixture
    def small_cluster(self):
        """Small cluster with 3 customers."""
        return pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [40.7128, 40.7260, 40.7489],
            'Longitude': [-74.0060, -73.9897, -73.9680]
        })
    
    @pytest.fixture
    def medium_cluster(self):
        """Medium cluster with 10 customers."""
        np.random.seed(42)
        n = 10
        return pd.DataFrame({
            'Customer_ID': [f'C{i}' for i in range(n)],
            'Latitude': 40.7 + np.random.uniform(-0.1, 0.1, n),
            'Longitude': -74.0 + np.random.uniform(-0.1, 0.1, n)
        })
    
    @pytest.fixture
    def large_cluster(self):
        """Large cluster with 25 customers."""
        np.random.seed(42)
        n = 25
        return pd.DataFrame({
            'Customer_ID': [f'C{i}' for i in range(n)],
            'Latitude': 40.7 + np.random.uniform(-0.2, 0.2, n),
            'Longitude': -74.0 + np.random.uniform(-0.2, 0.2, n)
        })
    
    @pytest.fixture
    def depot(self):
        """NYC depot location."""
        return {'latitude': 40.7128, 'longitude': -74.0060}
    
    def test_legacy_method_small_cluster(self, small_cluster, depot):
        """Test Legacy method with small cluster."""
        time, sequence = estimate_route_time(
            small_cluster, depot, 
            service_time=15,  # 15 minutes per customer
            avg_speed=25,     # 25 km/h in city
            method='Legacy'
        )
        
        # Legacy: 1 hour + service time
        # Expected: 1 + (3 * 15/60) = 1.75 hours
        assert 1.0 <= time <= 3.0  # Flexible range
        assert sequence == []  # Legacy doesn't provide sequence
    
    def test_legacy_method_large_cluster(self, large_cluster, depot):
        """Test Legacy method with large cluster."""
        time, sequence = estimate_route_time(
            large_cluster, depot,
            service_time=10,  # 10 minutes per customer
            avg_speed=30,
            method='Legacy'
        )
        
        # Legacy: 1 hour + service time
        # Expected: 1 + (25 * 10/60) ≈ 5.17 hours
        assert 3.0 <= time <= 8.0  # Flexible range
        assert sequence == []
    
    def test_bhh_method_small_cluster(self, small_cluster, depot):
        """Test BHH method with small cluster."""
        time, sequence = estimate_route_time(
            small_cluster, depot,
            service_time=15,
            avg_speed=25,
            method='BHH'
        )
        
        # BHH considers actual geography
        # For 3 customers in NYC area, expect 1-4 hours
        assert 0.5 <= time <= 4.0
        assert sequence == []  # BHH doesn't provide sequence
    
    def test_bhh_method_medium_cluster(self, medium_cluster, depot):
        """Test BHH method with medium cluster."""
        time, sequence = estimate_route_time(
            medium_cluster, depot,
            service_time=12,
            avg_speed=28,
            method='BHH'
        )
        
        # For 10 customers spread over ~20km area
        assert 1.0 <= time <= 6.0
        assert sequence == []
    
    def test_bhh_method_large_cluster(self, large_cluster, depot):
        """Test BHH method with large cluster."""
        time, sequence = estimate_route_time(
            large_cluster, depot,
            service_time=10,
            avg_speed=30,
            method='BHH'
        )
        
        # For 25 customers spread over ~40km area  
        assert 3.0 <= time <= 10.0
        assert sequence == []
    
    def test_tsp_method_small_cluster(self, small_cluster, depot):
        """Test TSP method with small cluster."""
        time, sequence = estimate_route_time(
            small_cluster, depot,
            service_time=15,
            avg_speed=25,
            method='TSP',
            max_route_time=8.0
        )
        
        # TSP gives optimal route
        assert 0.5 <= time <= 4.0
        assert len(sequence) == 5  # Depot -> 3 customers -> Depot
        assert sequence[0] == "Depot"
        assert sequence[-1] == "Depot"
        assert all(c in sequence for c in ['C1', 'C2', 'C3'])
    
    def test_tsp_method_medium_cluster(self, medium_cluster, depot):
        """Test TSP method with medium cluster."""
        time, sequence = estimate_route_time(
            medium_cluster, depot,
            service_time=12,
            avg_speed=28,
            method='TSP',
            max_route_time=10.0
        )
        
        # TSP should find efficient route
        assert 2.0 <= time <= 8.0
        assert len(sequence) == 12  # Depot + 10 customers + Depot
        assert sequence[0] == "Depot"
        assert sequence[-1] == "Depot"
    
    def test_tsp_method_with_pruning(self, large_cluster, depot):
        """Test TSP method with pruning for large cluster."""
        time, sequence = estimate_route_time(
            large_cluster, depot,
            service_time=15,  # Long service time
            avg_speed=20,     # Slow speed
            method='TSP',
            max_route_time=5.0,  # Tight constraint
            prune_tsp=True
        )
        
        # With pruning and tight constraint, might skip TSP
        assert 4.0 <= time <= 12.0
        # Sequence might be empty if pruned
        assert len(sequence) == 0 or len(sequence) == 27
    
    def test_tsp_without_max_time(self, small_cluster, depot):
        """Test TSP method without max_route_time constraint."""
        time, sequence = estimate_route_time(
            small_cluster, depot,
            service_time=20,
            avg_speed=20,
            method='TSP',
            max_route_time=None  # No limit
        )
        
        # Should still solve
        assert 0.5 <= time <= 6.0
        assert len(sequence) == 5
    
    def test_method_comparison_same_cluster(self, medium_cluster, depot):
        """Compare all three methods on same cluster."""
        params = {
            'service_time': 10,
            'avg_speed': 30
        }
        
        # Legacy
        legacy_time, _ = estimate_route_time(
            medium_cluster, depot,
            method='Legacy',
            **params
        )
        
        # BHH
        bhh_time, _ = estimate_route_time(
            medium_cluster, depot,
            method='BHH',
            **params
        )
        
        # TSP
        tsp_time, tsp_seq = estimate_route_time(
            medium_cluster, depot,
            method='TSP',
            max_route_time=10.0,
            **params
        )
        
        # All should be positive
        assert legacy_time > 0
        assert bhh_time > 0
        assert tsp_time > 0
        
        # TSP should have sequence
        assert len(tsp_seq) == 12
        
        # Generally: TSP <= BHH <= Legacy (but not always)
        # Just check they're in reasonable ranges
        assert 1.0 <= legacy_time <= 5.0
        assert 1.0 <= bhh_time <= 6.0  
        assert 1.0 <= tsp_time <= 6.0
    
    def test_edge_case_single_customer(self, depot):
        """Test all methods with single customer."""
        single_customer = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [40.7589],
            'Longitude': [-73.9851]
        })
        
        for method in ['Legacy', 'BHH', 'TSP']:
            time, sequence = estimate_route_time(
                single_customer, depot,
                service_time=30,
                avg_speed=25,
                method=method,
                max_route_time=5.0 if method == 'TSP' else None
            )
            
            # All methods should handle single customer
            assert 0.2 <= time <= 2.0
            
            if method == 'TSP':
                assert sequence == ["Depot", "C1", "Depot"]
            else:
                assert sequence == []
    
    def test_edge_case_zero_service_time(self, small_cluster, depot):
        """Test all methods with zero service time."""
        for method in ['Legacy', 'BHH', 'TSP']:
            time, sequence = estimate_route_time(
                small_cluster, depot,
                service_time=0,  # No service time
                avg_speed=40,
                method=method,
                max_route_time=5.0 if method == 'TSP' else None
            )
            
            # Should still have travel time
            if method == 'Legacy':
                assert time == 1.0  # Legacy always adds 1 hour
            else:
                assert 0.1 <= time <= 3.0
    
    def test_different_speeds(self, medium_cluster, depot):
        """Test impact of different speeds on all methods."""
        speeds = [20, 40, 60]  # Slow, medium, fast
        
        for method in ['Legacy', 'BHH', 'TSP']:
            times = []
            for speed in speeds:
                time, _ = estimate_route_time(
                    medium_cluster, depot,
                    service_time=10,
                    avg_speed=speed,
                    method=method,
                    max_route_time=10.0 if method == 'TSP' else None
                )
                times.append(time)
            
            # Faster speed should generally mean less time
            # (except Legacy which ignores speed)
            if method != 'Legacy':
                # Allow some tolerance for optimization variance
                assert times[0] >= times[2] * 0.8  # Slow >= Fast * 0.8
    
    def test_very_spread_cluster(self, depot):
        """Test methods with geographically spread cluster."""
        # Customers spread across large area
        spread_cluster = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3', 'C4'],
            'Latitude': [40.5, 40.9, 40.7, 40.3],    # ~60km spread
            'Longitude': [-74.3, -73.7, -74.0, -74.2]
        })
        
        for method in ['Legacy', 'BHH', 'TSP']:
            time, _ = estimate_route_time(
                spread_cluster, depot,
                service_time=15,
                avg_speed=50,  # Highway speed
                method=method,
                max_route_time=12.0 if method == 'TSP' else None
            )
            
            # Should take longer due to distances
            if method == 'Legacy':
                assert 1.5 <= time <= 3.0
            else:
                assert 2.0 <= time <= 8.0 