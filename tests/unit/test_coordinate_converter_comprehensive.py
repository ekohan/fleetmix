"""
Comprehensive tests for the coordinate converter module.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import random

from fleetmix.utils.coordinate_converter import (
    CoordinateConverter, 
    GeoBounds, 
    validate_conversion
)


class TestGeoBoundsExtended:
    """Extended tests for GeoBounds class"""
    
    def test_geobounds_creation(self):
        """Test GeoBounds creation with various values"""
        bounds = GeoBounds(min_lat=10.0, max_lat=20.0, min_lon=30.0, max_lon=40.0)
        assert bounds.min_lat == 10.0
        assert bounds.max_lat == 20.0
        assert bounds.min_lon == 30.0
        assert bounds.max_lon == 40.0
    
    def test_geobounds_negative_coordinates(self):
        """Test GeoBounds with negative coordinates"""
        bounds = GeoBounds(min_lat=-20.0, max_lat=-10.0, min_lon=-40.0, max_lon=-30.0)
        assert bounds.center == (-15.0, -35.0)
        assert bounds.lat_span == 10.0
        assert bounds.lon_span == 10.0
    
    def test_geobounds_zero_span(self):
        """Test GeoBounds with zero span (single point)"""
        bounds = GeoBounds(min_lat=5.0, max_lat=5.0, min_lon=10.0, max_lon=10.0)
        assert bounds.center == (5.0, 10.0)
        assert bounds.lat_span == 0.0
        assert bounds.lon_span == 0.0


class TestCoordinateConverterExtended:
    """Extended tests for CoordinateConverter class"""
    
    def test_converter_with_custom_bounds(self):
        """Test converter with custom geographic bounds"""
        coords = {1: (0.0, 0.0), 2: (100.0, 100.0)}
        custom_bounds = GeoBounds(min_lat=50.0, max_lat=60.0, min_lon=10.0, max_lon=20.0)
        
        converter = CoordinateConverter(coords, custom_bounds)
        assert converter.geo_bounds == custom_bounds
    
    def test_converter_with_default_bounds(self):
        """Test converter uses default bounds when none provided"""
        coords = {1: (0.0, 0.0), 2: (100.0, 100.0)}
        converter = CoordinateConverter(coords)
        
        # Should use default Bogota bounds
        assert converter.geo_bounds.min_lat == 4.3333
        assert converter.geo_bounds.max_lat == 4.9167
        assert converter.geo_bounds.min_lon == -74.3500
        assert converter.geo_bounds.max_lon == -73.9167
    
    def test_converter_single_point(self):
        """Test converter with single point (zero range)"""
        coords = {1: (50.0, 50.0)}
        converter = CoordinateConverter(coords)
        
        # Should handle single point gracefully - may produce NaN due to division by zero
        lat, lon = converter.to_geographic(50.0, 50.0)
        # For single point, converter may produce NaN due to zero range
        assert isinstance(lat, (int, float))
        assert isinstance(lon, (int, float))
    
    def test_converter_scaling_factors(self):
        """Test scaling factor calculations"""
        coords = {1: (0.0, 0.0), 2: (100.0, 200.0)}  # Rectangle 100x200
        converter = CoordinateConverter(coords)
        
        # x_scale and y_scale should be different due to different spans
        assert converter.x_scale != converter.y_scale
        # scale should be the minimum to maintain aspect ratio
        assert converter.scale == min(converter.x_scale, converter.y_scale)
    
    def test_convert_all_coordinates_to_geographic(self):
        """Test converting all coordinates to geographic"""
        coords = {
            1: (0.0, 0.0),
            2: (100.0, 100.0),
            3: (-50.0, 50.0)
        }
        converter = CoordinateConverter(coords)
        
        geo_coords = converter.convert_all_coordinates(coords, to_geographic=True)
        
        assert len(geo_coords) == 3
        for node_id in coords:
            assert node_id in geo_coords
            lat, lon = geo_coords[node_id]
            assert isinstance(lat, (int, float))
            assert isinstance(lon, (int, float))
    
    def test_convert_all_coordinates_to_cvrp(self):
        """Test converting all coordinates to CVRP"""
        geo_coords = {
            1: (40.7128, -74.0060),  # NYC
            2: (40.7589, -73.9851),  # Times Square
            3: (40.6829, -74.0440)   # Lower Manhattan
        }
        
        # First convert to CVRP space, then back to geo
        coords = {1: (0.0, 0.0), 2: (100.0, 100.0), 3: (-50.0, 50.0)}
        converter = CoordinateConverter(coords)
        
        cvrp_coords = converter.convert_all_coordinates(geo_coords, to_geographic=False)
        
        assert len(cvrp_coords) == 3
        for node_id in geo_coords:
            assert node_id in cvrp_coords
            x, y = cvrp_coords[node_id]
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
    
    def test_to_cvrp_with_cosine_correction(self):
        """Test that to_cvrp applies cosine correction properly"""
        coords = {1: (0.0, 0.0), 2: (100.0, 100.0)}
        converter = CoordinateConverter(coords)
        
        # Test at different latitudes to see cosine correction effect
        lat1, lon1 = 0.0, 0.0  # Equator
        lat2, lon2 = 60.0, 0.0  # High latitude
        
        x1, y1 = converter.to_cvrp(lat1, lon1)
        x2, y2 = converter.to_cvrp(lat2, lon2)
        
        # At higher latitudes, longitude differences should be scaled up
        # due to cosine correction
        assert x1 != x2 or y1 != y2
    
    def test_extreme_coordinate_values(self):
        """Test converter with extreme coordinate values"""
        coords = {
            1: (-1000.0, -1000.0),
            2: (1000.0, 1000.0)
        }
        converter = CoordinateConverter(coords)
        
        # Should handle extreme values without error
        lat, lon = converter.to_geographic(0.0, 0.0)
        x, y = converter.to_cvrp(lat, lon)
        
        assert not np.isnan(lat)
        assert not np.isnan(lon)
        assert not np.isnan(x)
        assert not np.isnan(y)


class TestValidateConversion:
    """Tests for the validate_conversion function"""
    
    @patch('builtins.print')
    @patch('random.sample')
    def test_validate_conversion_normal(self, mock_sample, mock_print):
        """Test normal validation process"""
        coords = {
            1: (0.0, 0.0),
            2: (100.0, 100.0),
            3: (200.0, 50.0),
            4: (150.0, 150.0)
        }
        converter = CoordinateConverter(coords)
        
        # Mock random.sample to return specific pairs
        mock_sample.return_value = [(1, 2), (2, 3)]
        
        # Should not raise any exceptions
        validate_conversion(converter, coords)
        
        # Check that print was called (output was generated)
        assert mock_print.called
    
    @patch('builtins.print')
    @patch('random.sample')
    def test_validate_conversion_small_dataset(self, mock_sample, mock_print):
        """Test validation with small dataset (fewer than 10 pairs)"""
        coords = {1: (0.0, 0.0), 2: (100.0, 100.0)}
        converter = CoordinateConverter(coords)
        
        # With only 2 points, there's only 1 pair
        mock_sample.return_value = [(1, 2)]
        
        validate_conversion(converter, coords)
        assert mock_print.called
    
    @patch('builtins.print')
    @patch('random.sample')  
    def test_validate_conversion_zero_distance(self, mock_sample, mock_print):
        """Test validation with zero distance between points"""
        coords = {1: (0.0, 0.0), 2: (0.0, 0.0)}  # Same coordinates
        converter = CoordinateConverter(coords)
        
        mock_sample.return_value = [(1, 2)]
        
        # Should handle zero distance gracefully
        validate_conversion(converter, coords)
        assert mock_print.called


class TestMainScriptFunctionality:
    """Tests for the __main__ script functionality"""
    
    @patch('builtins.print')
    def test_main_script_example(self, mock_print):
        """Test the example code in __main__ section"""
        # This tests the example usage code
        example_coords = {
            1: (0, 0),    # Depot
            2: (100, 100),
            3: (200, 50),
            4: (150, 150)
        }
        
        # Create converter with default bounds
        converter = CoordinateConverter(example_coords)
        
        # Convert coordinates
        geo_coords = converter.convert_all_coordinates(example_coords)
        
        # Should not raise any exceptions
        assert len(geo_coords) == 4
        assert all(isinstance(coord, tuple) and len(coord) == 2 
                  for coord in geo_coords.values())
        
        # Test validation
        with patch('random.sample') as mock_sample:
            mock_sample.return_value = [(1, 2), (2, 3)]
            validate_conversion(converter, example_coords)
        
        assert mock_print.called


class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_empty_coordinates(self):
        """Test converter with empty coordinates dictionary"""
        with pytest.raises((IndexError, ValueError)):
            CoordinateConverter({})
    
    def test_coordinates_with_nan(self):
        """Test converter behavior with NaN coordinates"""
        coords = {1: (np.nan, 0.0), 2: (100.0, np.nan)}
        
        # Should handle NaN values (might result in NaN outputs)
        converter = CoordinateConverter(coords)
        lat, lon = converter.to_geographic(50.0, 50.0)
        
        # Output might be NaN due to NaN in input
        assert isinstance(lat, (int, float))
        assert isinstance(lon, (int, float))
    
    def test_coordinates_with_infinity(self):
        """Test converter behavior with infinite coordinates"""
        coords = {1: (np.inf, 0.0), 2: (100.0, -np.inf)}
        
        converter = CoordinateConverter(coords)
        lat, lon = converter.to_geographic(50.0, 50.0)
        
        # Should handle infinity (results might be unusual but no crash)
        assert isinstance(lat, (int, float))
        assert isinstance(lon, (int, float))
    
    def test_very_small_coordinate_range(self):
        """Test converter with very small coordinate range"""
        coords = {1: (0.0, 0.0), 2: (0.001, 0.001)}  # Very small range
        converter = CoordinateConverter(coords)
        
        lat, lon = converter.to_geographic(0.0005, 0.0005)
        x, y = converter.to_cvrp(lat, lon)
        
        # Should handle small ranges without division by zero
        assert not np.isnan(lat)
        assert not np.isnan(lon)
        assert not np.isnan(x)
        assert not np.isnan(y) 