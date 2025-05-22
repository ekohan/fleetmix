import pandas as pd
import pytest
from fleetmix.utils.route_time import _legacy_estimation, _bhh_estimation, estimate_route_time


def test_legacy_estimation_edge_cases():
    # Zero customers yields 1 hour
    assert _legacy_estimation(0, service_time=30) == pytest.approx(1.0)
    # Two customers at 30 min each: 1 + 2*0.5 = 2 hours
    assert _legacy_estimation(2, service_time=30) == pytest.approx(2.0)


def test_bhh_estimation_bounds():
    # Build a small cluster of 3 points
    df = pd.DataFrame({
        'Latitude': [0.0, 0.0, 1.0],
        'Longitude': [0.0, 1.0, 0.0]
    })
    service_time = 30
    avg_speed = 60
    max_route_time = 10
    # Compute BHH and legacy
    t_bhh = _bhh_estimation(df, {'latitude':0,'longitude':0}, service_time, avg_speed)
    t_legacy = _legacy_estimation(len(df), service_time)
    # BHH >= legacy
    assert t_bhh >= t_legacy
    # BHH not ridiculously large: bounded by legacy + 2*max_route_time
    assert t_bhh <= t_legacy + 2*max_route_time

@pytest.mark.parametrize("method", ["Legacy", "BHH"])
def test_estimate_route_time_dispatch(method):
    # Single customer cluster
    df = pd.DataFrame({'Latitude': [0.0], 'Longitude': [0.0]})
    t, seq = estimate_route_time(
        df, 
        {'latitude':0,'longitude':0}, 
        service_time=15, 
        avg_speed=30, 
        method=method,
        prune_tsp=False
    )
    assert isinstance(t, float)
    assert isinstance(seq, list)
    # Legacy should return empty sequence
    if method == 'Legacy':
        assert seq == []
    # BHH returns empty too for singleton 