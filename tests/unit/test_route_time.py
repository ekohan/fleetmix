import pytest
import pandas as pd
import numpy as np

from fleetmix.utils.route_time import _legacy_estimation, _bhh_estimation


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