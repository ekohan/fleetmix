import pytest

from fleetmix.utils.coordinate_converter import CoordinateConverter, GeoBounds


def test_geobounds_center_and_spans():
    bounds = GeoBounds(min_lat=0.0, max_lat=10.0, min_lon=20.0, max_lon=30.0)
    # Center is average
    assert bounds.center == (5.0, 25.0)
    # Spans
    assert bounds.lat_span == 10.0
    assert bounds.lon_span == 10.0


def test_coordinate_round_trip():
    # Define simple CVRP coordinates
    coords = {
        1: (0.0, 0.0),
        2: (100.0, 200.0),
        3: (-50.0, 50.0),
    }
    converter = CoordinateConverter(coords)

    for node_id, (x, y) in coords.items():
        lat, lon = converter.to_geographic(x, y)
        x_rt, y_rt = converter.to_cvrp(lat, lon)
        # Allow small numerical deviation due to projection scaling (~0.1%)
        # Use both relative and absolute tolerance
        assert x_rt == pytest.approx(x, rel=1e-3, abs=1e-1)
        assert y_rt == pytest.approx(y, rel=1e-3, abs=1e-1)


def test_scale_maintains_aspect_ratio():
    # Create rectangle coords
    coords = {
        1: (0.0, 0.0),
        2: (0.0, 1.0),
        3: (2.0, 0.0),
    }
    converter = CoordinateConverter(coords)
    # The scale factor should be the minimum of x_scale and y_scale
    assert converter.scale == pytest.approx(
        min(converter.x_scale, converter.y_scale), rel=1e-6
    )
    # scale is no greater than each individual scale
    assert converter.scale <= converter.x_scale
    assert converter.scale <= converter.y_scale
