"""Unit tests for the route_time module."""

import unittest

import pandas as pd

from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    estimate_route_time,
)


class TestRouteTime(unittest.TestCase):
    """Test cases for route time estimation functions."""

    def test_calculate_total_service_time_hours(self):
        """Test service time calculation."""
        # Test normal case
        result = calculate_total_service_time_hours(5, 10)  # 5 customers, 10 min each
        self.assertEqual(result, 50 / 60)  # 50 minutes = 0.833... hours

        # Test zero customers
        result = calculate_total_service_time_hours(0, 10)
        self.assertEqual(result, 0.0)

        # Test zero service time
        result = calculate_total_service_time_hours(5, 0)
        self.assertEqual(result, 0.0)

    def test_legacy_estimation(self):
        """Test legacy estimation method through estimate_route_time."""
        # Create dummy customer data
        customers_df = pd.DataFrame(
            {
                "Customer_ID": ["C1", "C2"],
                "Latitude": [0.1, 0.2],
                "Longitude": [0.1, 0.2],
            }
        )
        depot = {"latitude": 0.0, "longitude": 0.0}

        # 2 customers, 30 min service time each
        time, sequence = estimate_route_time(
            customers_df, depot, 30, 30, method="Legacy"
        )

        # Legacy: 1 hour travel + 1 hour service (2 * 30 min)
        expected = 1 + (2 * 30 / 60)
        self.assertEqual(time, expected)
        self.assertEqual(sequence, [])

    def test_bhh_estimation(self):
        """Test BHH estimation method through estimate_route_time."""
        # Create dummy customer data
        customers_df = pd.DataFrame(
            {
                "Customer_ID": ["C1", "C2", "C3"],
                "Latitude": [0.1, 0.15, 0.2],
                "Longitude": [0.1, 0.15, 0.2],
            }
        )
        depot = {"latitude": 0.0, "longitude": 0.0}

        time, sequence = estimate_route_time(customers_df, depot, 10, 30, method="BHH")

        # BHH should return positive time
        self.assertGreater(time, 0)
        self.assertEqual(sequence, [])

    def test_invalid_method(self):
        """Test with invalid estimation method."""
        customers_df = pd.DataFrame(
            {"Customer_ID": ["C1"], "Latitude": [0.1], "Longitude": [0.1]}
        )
        depot = {"latitude": 0.0, "longitude": 0.0}

        with self.assertRaises(ValueError) as cm:
            estimate_route_time(customers_df, depot, 10, 30, method="INVALID")

        self.assertIn("Unknown route time estimation method", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
