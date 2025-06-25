from fleetmix.utils.coordinate_converter import CoordinateConverter, validate_conversion


def test_validate_conversion_roundtrip(capfd):
    coords = {
        1: (0.0, 0.0),
        2: (100.0, 50.0),
        3: (200.0, 150.0),
    }
    converter = CoordinateConverter(coords)
    # ensure convert back roundtrip roughly equal for depot node
    lat, lon = converter.to_geographic(*coords[1])
    x, y = converter.to_cvrp(lat, lon)
    assert abs(x - coords[1][0]) < 1e-1
    assert abs(y - coords[1][1]) < 1e-1

    # run verbose validation to hit lines 204-226
    validate_conversion(converter, coords)
    # consume captured stdout so pytest -q remains clean
    capfd.readouterr() 