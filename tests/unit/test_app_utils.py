"""Test app.py utility functions"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fleetmix.app import _get_available_instances, _list_instances, _setup_logging_from_flags, console
from fleetmix.utils.logging import LogLevel, FleetmixLogger


class TestAppUtils:
    """Test utility functions from app.py"""
    
    def test_setup_logging_from_flags_verbose(self):
        """Test _setup_logging_from_flags with verbose flag"""
        _setup_logging_from_flags(verbose=True, quiet=False, debug=False)
        assert FleetmixLogger.get_level() == LogLevel.VERBOSE
    
    def test_setup_logging_from_flags_quiet(self):
        """Test _setup_logging_from_flags with quiet flag"""
        _setup_logging_from_flags(verbose=False, quiet=True, debug=False)
        assert FleetmixLogger.get_level() == LogLevel.QUIET
    
    def test_setup_logging_from_flags_debug(self):
        """Test _setup_logging_from_flags with debug flag"""
        _setup_logging_from_flags(verbose=False, quiet=False, debug=True)
        assert FleetmixLogger.get_level() == LogLevel.DEBUG
    
    def test_setup_logging_from_flags_normal(self):
        """Test _setup_logging_from_flags with no flags (normal mode)"""
        _setup_logging_from_flags(verbose=False, quiet=False, debug=False)
        assert FleetmixLogger.get_level() == LogLevel.NORMAL

    @patch('fleetmix.app.Path')
    def test_get_available_instances_mcvrp(self, MockPath):
        """Test _get_available_instances for MCVRP suite"""
        # Create mock file objects
        mock_file1 = MagicMock()
        mock_file1.stem = 'instance1'
        mock_file2 = MagicMock()
        mock_file2.stem = 'instance2'
        
        # Create a chain of mocks that mimics the path construction
        mock_mcvrp_dir = MagicMock()
        mock_mcvrp_dir.glob.return_value = [mock_file1, mock_file2]
        
        # Set up the path chain: Path(__file__).parent / "benchmarking" / "datasets" / "mcvrp"
        MockPath.return_value.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_mcvrp_dir
        
        result = _get_available_instances('mcvrp')
        
        assert result == ['instance1', 'instance2']
        mock_mcvrp_dir.glob.assert_called_once_with("*.dat")

    @patch('fleetmix.app.Path')
    def test_get_available_instances_cvrp(self, MockPath):
        """Test _get_available_instances for CVRP suite"""
        # Create mock file objects
        mock_file1 = MagicMock()
        mock_file1.stem = 'X-n32-k5'
        mock_file2 = MagicMock()
        mock_file2.stem = 'X-n43-k6'
        
        # Create a chain of mocks that mimics the path construction
        mock_cvrp_dir = MagicMock()
        mock_cvrp_dir.glob.return_value = [mock_file1, mock_file2]
        
        # Set up the path chain: Path(__file__).parent / "benchmarking" / "datasets" / "cvrp"
        MockPath.return_value.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_cvrp_dir
        
        result = _get_available_instances('cvrp')
        
        assert result == ['X-n32-k5', 'X-n43-k6']
        mock_cvrp_dir.glob.assert_called_once_with("X-n*.vrp")

    def test_get_available_instances_invalid_suite(self):
        """Test _get_available_instances with invalid suite"""
        # For invalid suite, the function should return an empty list without file access
        result = _get_available_instances('invalid_suite_name')
        assert result == []

    @patch('fleetmix.app.console.print')
    @patch('fleetmix.app._get_available_instances')
    def test_list_instances_no_instances(self, mock_get_instances, mock_print):
        """Test _list_instances when no instances are found"""
        mock_get_instances.return_value = []
        _list_instances('test_suite')
        mock_print.assert_any_call("[yellow]No instances found for TEST_SUITE[/yellow]")

    @patch('fleetmix.app.console.print')
    @patch('fleetmix.app.Table')
    @patch('fleetmix.app._get_available_instances')
    def test_list_instances_with_instances(self, mock_get_instances, mock_table_class, mock_print):
        """Test _list_instances displays a table of instances"""
        mock_get_instances.return_value = ['instanceA', 'instanceB']
        mock_table_instance = MagicMock()
        mock_table_class.return_value = mock_table_instance
        _list_instances('test_suite')
        mock_table_class.assert_called_once_with(title="Available TEST_SUITE Instances", show_header=True)
        mock_table_instance.add_column.assert_called_once_with("Instance", style="cyan")
        assert mock_table_instance.add_row.call_count == 2
        mock_print.assert_any_call(mock_table_instance) 