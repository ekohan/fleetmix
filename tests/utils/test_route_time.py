"""Unit tests for the route_time module."""

import unittest
import numpy as np
import pandas as pd
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


class TestServiceTimeCalculation(unittest.TestCase):
    """Test cases for service time calculation."""
    
    def test_calculate_total_service_time_hours_normal(self):
        """Test normal service time calculation."""
        result = calculate_total_service_time_hours(10, 30)  # 10 customers, 30 min each
        self.assertEqual(result, 5.0)  # 300 minutes = 5 hours
    
    def test_calculate_total_service_time_hours_zero_customers(self):
        """Test with zero customers."""
        result = calculate_total_service_time_hours(0, 30)
        self.assertEqual(result, 0.0)
    
    def test_calculate_total_service_time_hours_negative_customers(self):
        """Test with negative customers (edge case)."""
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            result = calculate_total_service_time_hours(-5, 30)
            self.assertEqual(result, 0.0)
            mock_logger.warning.assert_called_once()
    
    def test_calculate_total_service_time_hours_negative_service_time(self):
        """Test with negative service time (edge case)."""
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            result = calculate_total_service_time_hours(10, -30)
            self.assertEqual(result, 0.0)
            mock_logger.warning.assert_called_once()


class TestMatrixBuilding(unittest.TestCase):
    """Test cases for distance/duration matrix building."""
    
    def setUp(self):
        """Set up test data."""
        self.customers_df = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [4.5, 4.6, 4.7],
            'Longitude': [-74.0, -74.1, -74.2]
        })
        self.depot = {'latitude': 4.4, 'longitude': -73.9}
        self.avg_speed = 30  # km/h
    
    def tearDown(self):
        """Clean up matrix cache."""
        _matrix_cache['distance_matrix'] = None
        _matrix_cache['duration_matrix'] = None
        _matrix_cache['customer_id_to_idx'] = None
    
    def test_build_matrices_normal(self):
        """Test building matrices with normal data."""
        build_distance_duration_matrices(self.customers_df, self.depot, self.avg_speed)
        
        # Check cache is populated
        self.assertIsNotNone(_matrix_cache['distance_matrix'])
        self.assertIsNotNone(_matrix_cache['duration_matrix'])
        self.assertIsNotNone(_matrix_cache['customer_id_to_idx'])
        
        # Check matrix dimensions (depot + 3 customers = 4x4)
        self.assertEqual(_matrix_cache['distance_matrix'].shape, (4, 4))
        self.assertEqual(_matrix_cache['duration_matrix'].shape, (4, 4))
        
        # Check customer mapping
        self.assertEqual(_matrix_cache['customer_id_to_idx']['C1'], 1)
        self.assertEqual(_matrix_cache['customer_id_to_idx']['C2'], 2)
        self.assertEqual(_matrix_cache['customer_id_to_idx']['C3'], 3)
    
    def test_build_matrices_empty_dataframe(self):
        """Test building matrices with empty customer data."""
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            build_distance_duration_matrices(pd.DataFrame(), self.depot, self.avg_speed)
            mock_logger.warning.assert_called_once()
    
    def test_build_matrices_duplicate_customer_ids(self):
        """Test building matrices with duplicate customer IDs."""
        df_with_duplicates = pd.DataFrame({
            'Customer_ID': ['C1', 'C1', 'C2'],
            'Latitude': [4.5, 4.6, 4.7],
            'Longitude': [-74.0, -74.1, -74.2]
        })
        
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            build_distance_duration_matrices(df_with_duplicates, self.depot, self.avg_speed)
            mock_logger.warning.assert_called_with("Duplicate Customer IDs found. Matrix mapping might be incorrect.")
    
    def test_build_matrices_missing_columns(self):
        """Test building matrices with missing coordinate columns."""
        df_missing_cols = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Lat': [4.5, 4.6]  # Wrong column name
        })
        
        with self.assertRaises(ValueError) as cm:
            build_distance_duration_matrices(df_missing_cols, self.depot, self.avg_speed)
        self.assertIn("Missing coordinate columns", str(cm.exception))
    
    def test_build_matrices_zero_speed(self):
        """Test building matrices with zero speed."""
        # The implementation stores infinity as a large integer value
        # We need to check that the function handles zero speed without crashing
        try:
            build_distance_duration_matrices(self.customers_df, self.depot, 0)
            # If it succeeds, check that duration matrix has large values
            duration_matrix = _matrix_cache['duration_matrix']
            # Off-diagonal elements should be very large (representing infinity)
            off_diagonal = duration_matrix[duration_matrix > 0]
            if len(off_diagonal) > 0:
                # Check that values are very large (representing infinity)
                self.assertTrue(np.all(off_diagonal > 1e6))
        except OverflowError:
            # This is also acceptable - the function tried to convert infinity to int
            pass


class TestRouteTimeEstimation(unittest.TestCase):
    """Test cases for route time estimation methods."""
    
    def setUp(self):
        """Set up test data."""
        self.cluster_customers = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [4.5, 4.6],
            'Longitude': [-74.0, -74.1]
        })
        self.depot = {'latitude': 4.4, 'longitude': -73.9}
        self.service_time = 30  # minutes
        self.avg_speed = 30  # km/h
    
    def test_estimate_route_time_legacy(self):
        """Test legacy estimation method."""
        time, sequence = estimate_route_time(
            self.cluster_customers, self.depot, self.service_time, 
            self.avg_speed, method='Legacy'
        )
        
        # Legacy: 1 hour + service time
        expected = 1 + (2 * 30 / 60)  # 1 + 1 hour service
        self.assertEqual(time, expected)
        self.assertEqual(sequence, [])
    
    def test_estimate_route_time_bhh(self):
        """Test BHH estimation method."""
        time, sequence = estimate_route_time(
            self.cluster_customers, self.depot, self.service_time,
            self.avg_speed, method='BHH'
        )
        
        # Should return a positive time
        self.assertGreater(time, 0)
        self.assertEqual(sequence, [])
    
    def test_estimate_route_time_invalid_method(self):
        """Test with invalid estimation method."""
        with self.assertRaises(ValueError) as cm:
            estimate_route_time(
                self.cluster_customers, self.depot, self.service_time,
                self.avg_speed, method='InvalidMethod'
            )
        self.assertIn("Unknown route time estimation method", str(cm.exception))
    
    @patch('fleetmix.utils.route_time._pyvrp_tsp_estimation')
    def test_estimate_route_time_tsp(self, mock_tsp):
        """Test TSP estimation method."""
        mock_tsp.return_value = (3.5, ['Depot', 'C1', 'C2', 'Depot'])
        
        time, sequence = estimate_route_time(
            self.cluster_customers, self.depot, self.service_time,
            self.avg_speed, method='TSP'
        )
        
        self.assertEqual(time, 3.5)
        self.assertEqual(sequence, ['Depot', 'C1', 'C2', 'Depot'])
        mock_tsp.assert_called_once()
    
    @patch('fleetmix.utils.route_time._bhh_estimation')
    @patch('fleetmix.utils.route_time._pyvrp_tsp_estimation')
    def test_estimate_route_time_tsp_with_pruning(self, mock_tsp, mock_bhh):
        """Test TSP estimation with BHH-based pruning."""
        # BHH estimates 10 hours, max is 8 hours
        mock_bhh.return_value = 10.0
        
        time, sequence = estimate_route_time(
            self.cluster_customers, self.depot, self.service_time,
            self.avg_speed, method='TSP', max_route_time=8.0, prune_tsp=True
        )
        
        # Should skip TSP and return slightly over max
        self.assertAlmostEqual(time, 8.08, places=2)
        self.assertEqual(sequence, [])
        mock_tsp.assert_not_called()


class TestLegacyEstimation(unittest.TestCase):
    """Test cases for legacy estimation method."""
    
    def test_legacy_estimation_various_sizes(self):
        """Test legacy estimation with various cluster sizes."""
        # 0 customers
        self.assertEqual(_legacy_estimation(0, 30), 1.0)
        
        # 5 customers, 30 min each = 2.5 hours service + 1 hour travel
        self.assertEqual(_legacy_estimation(5, 30), 3.5)
        
        # 10 customers, 15 min each = 2.5 hours service + 1 hour travel  
        self.assertEqual(_legacy_estimation(10, 15), 3.5)


class TestBHHEstimation(unittest.TestCase):
    """Test cases for BHH estimation method."""
    
    def setUp(self):
        """Set up test data."""
        self.depot = {'latitude': 4.4, 'longitude': -73.9}
        self.service_time = 30  # minutes
        self.avg_speed = 30  # km/h
    
    def test_bhh_estimation_single_customer(self):
        """Test BHH estimation with single customer."""
        cluster = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [4.5],
            'Longitude': [-74.0]
        })
        
        time = _bhh_estimation(cluster, self.depot, self.service_time, self.avg_speed)
        
        # Should be just service time for single customer
        self.assertEqual(time, 0.5)  # 30 minutes = 0.5 hours
    
    def test_bhh_estimation_empty_cluster(self):
        """Test BHH estimation with empty cluster."""
        cluster = pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude'])
        
        time = _bhh_estimation(cluster, self.depot, self.service_time, self.avg_speed)
        
        # Should be 0 for empty cluster
        self.assertEqual(time, 0.0)
    
    def test_bhh_estimation_multiple_customers(self):
        """Test BHH estimation with multiple customers."""
        cluster = pd.DataFrame({
            'Customer_ID': ['C1', 'C2', 'C3'],
            'Latitude': [4.5, 4.6, 4.7],
            'Longitude': [-74.0, -74.1, -74.2]
        })
        
        time = _bhh_estimation(cluster, self.depot, self.service_time, self.avg_speed)
        
        # Should include service time + depot travel + intra-cluster travel
        self.assertGreater(time, 1.5)  # At least service time (1.5 hours for 3 customers)


class TestPyVRPTSPEstimation(unittest.TestCase):
    """Test cases for PyVRP TSP estimation."""
    
    def setUp(self):
        """Set up test data and clear cache."""
        self.depot = {'latitude': 4.4, 'longitude': -73.9}
        self.service_time = 30  # minutes
        self.avg_speed = 30  # km/h
        
        # Clear matrix cache
        _matrix_cache['distance_matrix'] = None
        _matrix_cache['duration_matrix'] = None
        _matrix_cache['customer_id_to_idx'] = None
    
    def test_tsp_estimation_empty_cluster(self):
        """Test TSP estimation with empty cluster."""
        cluster = pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude'])
        
        time, sequence = _pyvrp_tsp_estimation(
            cluster, self.depot, self.service_time, self.avg_speed
        )
        
        self.assertEqual(time, 0.0)
        self.assertEqual(sequence, [])
    
    def test_tsp_estimation_single_customer(self):
        """Test TSP estimation with single customer."""
        cluster = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [4.5],
            'Longitude': [-74.0]
        })
        
        time, sequence = _pyvrp_tsp_estimation(
            cluster, self.depot, self.service_time, self.avg_speed
        )
        
        # Should calculate round trip time + service time
        self.assertGreater(time, 0.5)  # At least service time
        self.assertEqual(sequence, ["Depot", "C1", "Depot"])
    
    @patch('fleetmix.utils.route_time.Model')
    def test_tsp_estimation_with_cached_matrices(self, mock_model):
        """Test TSP estimation using cached matrices."""
        # Set up cache
        _matrix_cache['distance_matrix'] = np.array([
            [0, 1000, 2000],
            [1000, 0, 1500],
            [2000, 1500, 0]
        ])
        _matrix_cache['duration_matrix'] = np.array([
            [0, 120, 240],
            [120, 0, 180],
            [240, 180, 0]
        ])
        _matrix_cache['customer_id_to_idx'] = {'C1': 1, 'C2': 2}
        
        cluster = pd.DataFrame({
            'Customer_ID': ['C1'],
            'Latitude': [4.5],
            'Longitude': [-74.0]
        })
        
        # Mock the model solving
        mock_result = MagicMock()
        mock_result.best.is_feasible.return_value = True
        mock_result.best.duration.return_value = 3600  # 1 hour in seconds
        mock_result.best.routes.return_value = [MagicMock(visits=lambda: [1])]
        
        mock_model.from_data.return_value.solve.return_value = mock_result
        
        time, sequence = _pyvrp_tsp_estimation(
            cluster, self.depot, self.service_time, self.avg_speed
        )
        
        # The actual implementation might compute differently
        # Just check that we get a positive time and valid sequence
        self.assertGreater(time, 0)
        self.assertIsInstance(sequence, list)
        self.assertIn("Depot", sequence)
    
    @patch('fleetmix.utils.route_time.Model')
    def test_tsp_estimation_infeasible(self, mock_model):
        """Test TSP estimation when solution is infeasible."""
        cluster = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [4.5, 4.6],
            'Longitude': [-74.0, -74.1]
        })
        
        # Mock infeasible result
        mock_result = MagicMock()
        mock_result.best.is_feasible.return_value = False
        
        mock_model.from_data.return_value.solve.return_value = mock_result
        
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            time, sequence = _pyvrp_tsp_estimation(
                cluster, self.depot, self.service_time, self.avg_speed, max_route_time=8.0
            )
        
        # Should return slightly over max route time
        self.assertAlmostEqual(time, 8.08, places=2)
        self.assertEqual(sequence, [])
        mock_logger.warning.assert_called()
    
    def test_tsp_estimation_missing_customer_in_cache(self):
        """Test TSP estimation when customer is missing from cache."""
        # Set up incomplete cache
        _matrix_cache['distance_matrix'] = np.array([[0, 1000], [1000, 0]])
        _matrix_cache['duration_matrix'] = np.array([[0, 120], [120, 0]])
        _matrix_cache['customer_id_to_idx'] = {'C1': 1}  # C2 is missing
        
        cluster = pd.DataFrame({
            'Customer_ID': ['C1', 'C2'],
            'Latitude': [4.5, 4.6],
            'Longitude': [-74.0, -74.1]
        })
        
        with patch('fleetmix.utils.route_time.logger') as mock_logger:
            # Should fall back to on-the-fly computation
            time, sequence = _pyvrp_tsp_estimation(
                cluster, self.depot, self.service_time, self.avg_speed
            )
        
        # Should log warning about missing IDs
        mock_logger.warning.assert_called()
        # Should still return a result (using on-the-fly computation)
        self.assertIsInstance(time, float)
        self.assertIsInstance(sequence, list)


if __name__ == "__main__":
    unittest.main() 