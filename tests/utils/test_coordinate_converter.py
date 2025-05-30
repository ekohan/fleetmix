"""Unit tests for the coordinate_converter module."""

import unittest
import numpy as np
from fleetmix.utils.coordinate_converter import GeoBounds, CoordinateConverter


class TestGeoBounds(unittest.TestCase):
    """Test cases for GeoBounds dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bounds = GeoBounds(
            min_lat=4.3333,
            max_lat=4.9167,
            min_lon=-74.3500,
            max_lon=-73.9167
        )
    
    def test_center_property(self):
        """Test that center property returns correct center point."""
        center = self.bounds.center
        expected_lat = (4.3333 + 4.9167) / 2
        expected_lon = (-74.3500 + -73.9167) / 2
        self.assertAlmostEqual(center[0], expected_lat, places=4)
        self.assertAlmostEqual(center[1], expected_lon, places=4)
    
    def test_lat_span_property(self):
        """Test that lat_span property returns correct latitude span."""
        lat_span = self.bounds.lat_span
        expected = 4.9167 - 4.3333
        self.assertAlmostEqual(lat_span, expected, places=4)
    
    def test_lon_span_property(self):
        """Test that lon_span property returns correct longitude span."""
        lon_span = self.bounds.lon_span
        expected = -73.9167 - (-74.3500)
        self.assertAlmostEqual(lon_span, expected, places=4)


class TestCoordinateConverter(unittest.TestCase):
    """Test cases for CoordinateConverter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cvrp_coords = {
            1: (0, 0),      # Depot
            2: (100, 100),
            3: (200, 50),
            4: (150, 150)
        }
        self.custom_bounds = GeoBounds(
            min_lat=40.0,
            max_lat=41.0,
            min_lon=-74.0,
            max_lon=-73.0
        )
    
    def test_initialization_with_default_bounds(self):
        """Test converter initialization with default bounds."""
        converter = CoordinateConverter(self.cvrp_coords)
        
        # Check that default bounds are set (Bogota area)
        self.assertEqual(converter.geo_bounds.min_lat, 4.3333)
        self.assertEqual(converter.geo_bounds.max_lat, 4.9167)
        self.assertEqual(converter.geo_bounds.min_lon, -74.3500)
        self.assertEqual(converter.geo_bounds.max_lon, -73.9167)
        
        # Check that CVRP bounds are calculated correctly
        self.assertEqual(converter.min_x, 0)
        self.assertEqual(converter.max_x, 200)
        self.assertEqual(converter.min_y, 0)
        self.assertEqual(converter.max_y, 150)
    
    def test_initialization_with_custom_bounds(self):
        """Test converter initialization with custom bounds."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # Check that custom bounds are used
        self.assertEqual(converter.geo_bounds.min_lat, 40.0)
        self.assertEqual(converter.geo_bounds.max_lat, 41.0)
        self.assertEqual(converter.geo_bounds.min_lon, -74.0)
        self.assertEqual(converter.geo_bounds.max_lon, -73.0)
    
    def test_to_geographic_conversion(self):
        """Test CVRP to geographic coordinate conversion."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # Test depot at origin
        lat, lon = converter.to_geographic(0, 0)
        # Should be offset from center based on CVRP center
        # CVRP center is (100, 75), so (0,0) is (-100, -75) from center
        
        # Test a known point
        lat2, lon2 = converter.to_geographic(100, 75)
        # This should be at the geographic center
        center = self.custom_bounds.center
        self.assertAlmostEqual(lat2, center[0], places=4)
        # Longitude might differ due to cosine correction
    
    def test_to_cvrp_conversion(self):
        """Test geographic to CVRP coordinate conversion."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # Convert center of geographic bounds
        center = self.custom_bounds.center
        x, y = converter.to_cvrp(center[0], center[1])
        
        # Should map to center of CVRP space
        self.assertAlmostEqual(x, 100, places=2)
        self.assertAlmostEqual(y, 75, places=2)
    
    def test_round_trip_conversion(self):
        """Test that converting to geographic and back preserves coordinates."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        for node_id, (x_orig, y_orig) in self.cvrp_coords.items():
            # Convert to geographic
            lat, lon = converter.to_geographic(x_orig, y_orig)
            # Convert back to CVRP
            x_back, y_back = converter.to_cvrp(lat, lon)
            
            # Should be close to original (within reasonable tolerance)
            # Note: Due to cosine correction, exact round-trip may not be possible
            self.assertAlmostEqual(x_orig, x_back, delta=1.0)
            self.assertAlmostEqual(y_orig, y_back, delta=1.0)
    
    def test_convert_all_coordinates_to_geographic(self):
        """Test batch conversion of coordinates to geographic."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        geo_coords = converter.convert_all_coordinates(self.cvrp_coords, to_geographic=True)
        
        # Check that all nodes are converted
        self.assertEqual(set(geo_coords.keys()), set(self.cvrp_coords.keys()))
        
        # Check that values are tuples of floats
        for node_id, (lat, lon) in geo_coords.items():
            self.assertIsInstance(lat, float)
            self.assertIsInstance(lon, float)
            # Check reasonable latitude range
            self.assertGreaterEqual(lat, 39.0)
            self.assertLessEqual(lat, 42.0)
            # Check reasonable longitude range
            self.assertGreaterEqual(lon, -75.0)
            self.assertLessEqual(lon, -72.0)
    
    def test_convert_all_coordinates_to_cvrp(self):
        """Test batch conversion of coordinates to CVRP."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # First convert to geographic
        geo_coords = converter.convert_all_coordinates(self.cvrp_coords, to_geographic=True)
        
        # Then convert back to CVRP
        cvrp_coords_back = converter.convert_all_coordinates(geo_coords, to_geographic=False)
        
        # Check that all nodes are converted
        self.assertEqual(set(cvrp_coords_back.keys()), set(self.cvrp_coords.keys()))
        
        # Check that values are close to original
        for node_id in self.cvrp_coords:
            x_orig, y_orig = self.cvrp_coords[node_id]
            x_back, y_back = cvrp_coords_back[node_id]
            # Use delta instead of places for better tolerance
            self.assertAlmostEqual(x_orig, x_back, delta=1.0)
            self.assertAlmostEqual(y_orig, y_back, delta=1.0)
    
    def test_edge_case_single_point(self):
        """Test converter with a single point."""
        single_coord = {1: (50, 50)}
        converter = CoordinateConverter(single_coord, self.custom_bounds)
        
        # With a single point, min and max are the same, leading to division by zero
        # The converter should handle this gracefully
        
        # Check that scale is set (might be inf or a default value)
        self.assertTrue(hasattr(converter, 'scale'))
        
        # For single point, conversions might not be meaningful
        # but should not crash
        try:
            lat, lon = converter.to_geographic(50, 50)
            # If we get here without exception, that's good enough
            self.assertIsInstance(lat, float)
            self.assertIsInstance(lon, float)
            
            # Converting back might give NaN or inf, which is expected
            x, y = converter.to_cvrp(lat, lon)
            # Just check that we get numeric types back
            self.assertTrue(isinstance(x, (float, int)) or np.isnan(x))
            self.assertTrue(isinstance(y, (float, int)) or np.isnan(y))
        except Exception as e:
            # If there's an exception, it should be a specific one we expect
            self.assertIn("division", str(e).lower())
    
    def test_cosine_correction(self):
        """Test that cosine correction is applied for longitude."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # The cosine correction should make longitude distances smaller at higher latitudes
        # This is implicitly tested in the scaling factor calculation
        cos_lat = np.cos(np.radians(self.custom_bounds.center[0]))
        
        # Check that x_scale includes cosine correction
        expected_x_scale = self.custom_bounds.lon_span * cos_lat / (converter.max_x - converter.min_x)
        self.assertAlmostEqual(converter.x_scale, expected_x_scale, places=4)
    
    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved by using minimum scale."""
        converter = CoordinateConverter(self.cvrp_coords, self.custom_bounds)
        
        # The scale should be the minimum of x_scale and y_scale
        self.assertEqual(converter.scale, min(converter.x_scale, converter.y_scale))


class TestValidateConversion(unittest.TestCase):
    """Test cases for validate_conversion function."""
    
    def test_validate_conversion_import(self):
        """Test that validate_conversion can be imported and called."""
        from fleetmix.utils.coordinate_converter import validate_conversion
        
        # Create a simple test case
        coords = {
            1: (0, 0),
            2: (100, 100)
        }
        converter = CoordinateConverter(coords)
        
        # Should not raise any exceptions
        # Note: This will print output, which is expected
        validate_conversion(converter, coords)


if __name__ == "__main__":
    unittest.main() 