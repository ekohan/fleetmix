"""Test the app module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fleetmix.app import (
    _get_available_instances,
    _setup_logging_from_flags
)


def test_get_available_instances_mcvrp():
    """Test getting available MCVRP instances."""
    instances = _get_available_instances("mcvrp")
    
    # Should return a list of instance names
    assert isinstance(instances, list)
    # MCVRP instances should be .dat files without extension
    if instances:  # Only test if instances exist
        assert all(not instance.endswith('.dat') for instance in instances)


def test_get_available_instances_cvrp():
    """Test getting available CVRP instances."""
    instances = _get_available_instances("cvrp")
    
    # Should return a list of instance names
    assert isinstance(instances, list)
    # CVRP instances should start with X-n
    if instances:  # Only test if instances exist
        assert all(instance.startswith('X-n') for instance in instances)


def test_get_available_instances_invalid_suite():
    """Test getting instances for invalid suite."""
    instances = _get_available_instances("invalid")
    
    # Should return empty list for invalid suite
    assert instances == []


@patch('fleetmix.app.setup_logging')
def test_setup_logging_from_flags_verbose(mock_setup_logging):
    """Test logging setup with verbose flag."""
    from fleetmix.utils.logging import LogLevel
    
    _setup_logging_from_flags(verbose=True, quiet=False, debug=False)
    
    mock_setup_logging.assert_called_once_with(LogLevel.VERBOSE)


@patch('fleetmix.app.setup_logging')
def test_setup_logging_from_flags_quiet(mock_setup_logging):
    """Test logging setup with quiet flag."""
    from fleetmix.utils.logging import LogLevel
    
    _setup_logging_from_flags(verbose=False, quiet=True, debug=False)
    
    mock_setup_logging.assert_called_once_with(LogLevel.QUIET)


@patch('fleetmix.app.setup_logging')
def test_setup_logging_from_flags_debug(mock_setup_logging):
    """Test logging setup with debug flag."""
    from fleetmix.utils.logging import LogLevel
    
    _setup_logging_from_flags(verbose=False, quiet=False, debug=True)
    
    mock_setup_logging.assert_called_once_with(LogLevel.DEBUG)


@patch('fleetmix.app.setup_logging')
def test_setup_logging_from_flags_default(mock_setup_logging):
    """Test logging setup with default flags."""
    _setup_logging_from_flags(verbose=False, quiet=False, debug=False)
    
    # Should call setup_logging() with no arguments (relies on default behavior)
    mock_setup_logging.assert_called_once_with()


@patch('fleetmix.app.setup_logging')
def test_setup_logging_from_flags_priority(mock_setup_logging):
    """Test logging setup flag priority (debug > verbose > quiet)."""
    from fleetmix.utils.logging import LogLevel
    
    # Debug overrides verbose and quiet
    _setup_logging_from_flags(verbose=True, quiet=True, debug=True)
    mock_setup_logging.assert_called_with(LogLevel.DEBUG)
    
    # Verbose overrides quiet (when debug is False)
    _setup_logging_from_flags(verbose=True, quiet=True, debug=False)
    mock_setup_logging.assert_called_with(LogLevel.VERBOSE) 