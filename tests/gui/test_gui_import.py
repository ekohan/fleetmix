"""
Test GUI module import functionality.
"""
import pytest


def test_gui_imports():
    """Test that GUI module can be imported."""
    try:
        from fleetmix import gui
        assert hasattr(gui, 'run_optimization_in_process')
        assert hasattr(gui, 'main')
    except ImportError:
        pytest.skip("GUI dependencies not installed") 