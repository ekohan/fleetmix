"""Test coordinate_converter.py main block execution"""
import pytest
from unittest.mock import patch
import subprocess
import sys


class TestCoordinateConverterMain:
    """Test the main block of coordinate_converter.py"""
    
    def test_main_block_execution(self):
        """Test the main block executes correctly when module is run directly"""
        # Run the module as a script and capture output
        result = subprocess.run(
            [sys.executable, '-m', 'fleetmix.utils.coordinate_converter'],
            capture_output=True,
            text=True,
            cwd='/Users/ekohan/MIT/fleetmix'
        )
        
        # Check the script ran successfully
        assert result.returncode == 0
        
        # Check output contains expected strings
        output = result.stdout
        assert "Original CVRP coordinates:" in output
        assert "Node 1: (0, 0)" in output
        assert "Node 2: (100, 100)" in output
        assert "Node 3: (200, 50)" in output
        assert "Node 4: (150, 150)" in output
        assert "Converted geographic coordinates:" in output
        
        # Check that validate_conversion output is present
        assert "Validating coordinate conversion:" in output
        assert "Comparing relative distances" in output 