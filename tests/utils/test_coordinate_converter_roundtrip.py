import math
import warnings

import numpy as np
import pytest

from fleetmix.utils.coordinate_converter import CoordinateConverter, GeoBounds


def _simple_coords():
    """Return a minimal coordinate set suitable for round-trip testing."""
    return {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (0.0, 10.0)}


def test_round_trip_preserves_distance():
    coords = _simple_coords()
    converter = CoordinateConverter(coords)

    # Round-trip each point and ensure small numeric error
    for node_id, (x, y) in coords.items():
        lat, lon = converter.to_geographic(x, y)
        x2, y2 = converter.to_cvrp(lat, lon)
        assert math.isclose(x, x2, rel_tol=1e-2, abs_tol=1e-2)
        assert math.isclose(y, y2, rel_tol=1e-2, abs_tol=1e-2)


def test_degenerate_span_avoids_crash():
    """All points share the same x-coordinate → x_span == 0; ensure no exception."""
    coords = {1: (0.0, 0.0), 2: (0.0, 5.0)}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        converter = CoordinateConverter(coords)
        lat, lon = converter.to_geographic(0.0, 0.0)
        assert isinstance(lat, float) and isinstance(lon, float) 