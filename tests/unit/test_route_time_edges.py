"""Edge cases for route time estimation."""

import pandas as pd

from fleetmix.utils.route_time import (
    calculate_total_service_time_hours,
    estimate_route_time,
)


def test_estimate_route_time_empty_dataframe():
    """Test with empty customer dataframe."""
    df = pd.DataFrame(columns=["Customer_ID", "Latitude", "Longitude"])
    depot = {"latitude": 0, "longitude": 0}

    # Legacy method with empty customers
    time, seq = estimate_route_time(df, depot, 30, 60, method="Legacy")
    assert time == 1.0  # Just the constant 1 hour
    assert seq == []

    # BHH method with empty customers
    time, seq = estimate_route_time(df, depot, 30, 60, method="BHH")
    assert time == 0.0  # No customers, no time
    assert seq == []


def test_estimate_route_time_single_customer():
    """Test with single customer."""
    df = pd.DataFrame({"Customer_ID": ["C1"], "Latitude": [0.1], "Longitude": [0.1]})
    depot = {"latitude": 0, "longitude": 0}

    # Legacy method
    time, seq = estimate_route_time(df, depot, 30, 60, method="Legacy")
    assert time == 1.5  # 1 hour + 30 min service
    assert seq == []

    # BHH method
    time, seq = estimate_route_time(df, depot, 30, 60, method="BHH")
    assert time == 0.5  # Just service time for single customer
    assert seq == []


def test_negative_service_time():
    """Test that negative service time is handled correctly."""
    # Should be handled by calculate_total_service_time_hours
    result = calculate_total_service_time_hours(5, -30)
    assert result == 0.0  # Negative service time returns 0


def test_negative_customers():
    """Test that negative number of customers is handled correctly."""
    result = calculate_total_service_time_hours(-5, 30)
    assert result == 0.0  # Negative customers returns 0
