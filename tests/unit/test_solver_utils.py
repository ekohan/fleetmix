"""Test the solver utilities module."""

import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from fleetmix.config.params import RuntimeParams
from fleetmix.utils.solver import (
    extract_optimality_gap,
    GurobiAdapter,
    CbcAdapter,
    pick_solver,
)


@pytest.fixture
def runtime_params(tmp_path):
    """Create basic runtime parameters for tests."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    return RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="auto",
        gap_rel=0.0,
        time_limit=None,
    )


def test_extract_optimality_gap_from_model_attribute():
    """Test extracting gap from model's solutionGap attribute."""
    model = Mock()
    model.solutionGap = 0.0048  # 0.48%
    solver = Mock()
    
    gap = extract_optimality_gap(model, solver)
    
    # Should convert to percentage
    assert gap == pytest.approx(0.48, abs=0.01)


def test_extract_optimality_gap_already_percentage():
    """Test gap extraction when value is already in percentage (lines 49-50)."""
    model = Mock()
    model.solutionGap = 5.5  # Already in percentage
    solver = Mock()
    
    gap = extract_optimality_gap(model, solver)
    
    assert gap == 5.5


def test_extract_optimality_gap_no_attribute():
    """Test gap extraction when model has no solutionGap attribute."""
    model = Mock(spec=[])  # No solutionGap attribute
    solver = Mock()
    
    gap = extract_optimality_gap(model, solver)
    
    # Should return None when unavailable
    assert gap is None


def test_gurobi_adapter_get_pulp_solver(runtime_params):
    """Test GurobiAdapter.get_pulp_solver."""
    adapter = GurobiAdapter()
    
    with patch("pulp.GUROBI_CMD") as MockGurobiCmd:
        mock_solver = Mock()
        MockGurobiCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(runtime_params)
        
        # Should create a Gurobi solver
        MockGurobiCmd.assert_called_once()
        assert solver == mock_solver


def test_gurobi_adapter_with_gap_rel(tmp_path):
    """Test GurobiAdapter with gap_rel parameter."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="gurobi",
        gap_rel=0.05,
        time_limit=None,
    )
    adapter = GurobiAdapter()
    
    with patch("pulp.GUROBI_CMD") as MockGurobiCmd:
        mock_solver = Mock()
        MockGurobiCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(params)
        
        # Should pass gapRel
        call_kwargs = MockGurobiCmd.call_args[1]
        assert "gapRel" in call_kwargs
        assert call_kwargs["gapRel"] == 0.05


def test_gurobi_adapter_with_time_limit(tmp_path):
    """Test GurobiAdapter with time_limit parameter."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="gurobi",
        gap_rel=0.0,
        time_limit=300,
    )
    adapter = GurobiAdapter()
    
    with patch("pulp.GUROBI_CMD") as MockGurobiCmd:
        mock_solver = Mock()
        MockGurobiCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(params)
        
        # Should pass TimeLimit in options
        call_kwargs = MockGurobiCmd.call_args[1]
        assert "options" in call_kwargs
        assert ("TimeLimit", 300) in call_kwargs["options"]


def test_gurobi_adapter_available():
    """Test GurobiAdapter.available property."""
    adapter = GurobiAdapter()
    
    # Availability depends on whether gurobipy is installed
    # Just check it returns a boolean
    assert isinstance(adapter.available, bool)


def test_cbc_adapter_get_pulp_solver(runtime_params):
    """Test CbcAdapter.get_pulp_solver."""
    adapter = CbcAdapter()
    
    with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
        mock_solver = Mock()
        MockCbcCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(runtime_params)
        
        # Should create a CBC solver
        MockCbcCmd.assert_called_once()
        assert solver == mock_solver


def test_cbc_adapter_with_gap_rel(tmp_path):
    """Test CbcAdapter with gap_rel parameter."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="cbc",
        gap_rel=0.05,
        time_limit=None,
    )
    adapter = CbcAdapter()
    
    with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
        mock_solver = Mock()
        MockCbcCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(params)
        
        # Should pass gapRel
        call_kwargs = MockCbcCmd.call_args[1]
        assert "gapRel" in call_kwargs
        assert call_kwargs["gapRel"] == 0.05


def test_cbc_adapter_with_time_limit(tmp_path):
    """Test CbcAdapter with time_limit parameter."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="cbc",
        gap_rel=0.0,
        time_limit=300,
    )
    adapter = CbcAdapter()
    
    with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
        mock_solver = Mock()
        MockCbcCmd.return_value = mock_solver
        
        solver = adapter.get_pulp_solver(params)
        
        # Should pass timeLimit
        call_kwargs = MockCbcCmd.call_args[1]
        assert "timeLimit" in call_kwargs
        assert call_kwargs["timeLimit"] == 300


def test_cbc_adapter_available():
    """Test CbcAdapter.available property (always True)."""
    adapter = CbcAdapter()
    
    # CBC is always available
    assert adapter.available is True


def test_pick_solver_gurobi_explicit(tmp_path, monkeypatch):
    """Test pick_solver with explicit gurobi choice."""
    monkeypatch.delenv("FSM_SOLVER", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="gurobi",
        gap_rel=0.0,
        time_limit=None,
    )
    
    with patch("pulp.GUROBI_CMD") as MockGurobiCmd:
        mock_solver = Mock()
        MockGurobiCmd.return_value = mock_solver
        
        solver = pick_solver(params)
        
        MockGurobiCmd.assert_called_once()


def test_pick_solver_cbc_explicit(tmp_path):
    """Test pick_solver with explicit cbc choice."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="cbc",
        gap_rel=0.0,
        time_limit=None,
    )
    
    with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
        mock_solver = Mock()
        MockCbcCmd.return_value = mock_solver
        
        solver = pick_solver(params)
        
        MockCbcCmd.assert_called_once()


def test_pick_solver_auto_gurobi_fails(tmp_path):
    """Test pick_solver with auto when Gurobi instantiation fails (lines 203-205)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="auto",
        gap_rel=0.0,
        time_limit=None,
    )
    
    # Mock Gurobi as available but failing on instantiation
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.return_value = Mock()  # Gurobi available
        
        with patch("pulp.GUROBI_CMD") as MockGurobiCmd:
            MockGurobiCmd.side_effect = OSError("Gurobi license error")
            
            with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
                mock_cbc_solver = Mock()
                MockCbcCmd.return_value = mock_cbc_solver
                
                solver = pick_solver(params)
                
                # Should fall back to CBC
                assert solver == mock_cbc_solver


def test_pick_solver_env_var_override(tmp_path, monkeypatch):
    """Test pick_solver with FSM_SOLVER environment variable."""
    monkeypatch.setenv("FSM_SOLVER", "cbc")
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# minimal config")
    
    params = RuntimeParams(
        config=config_file,
        verbose=False,
        debug=False,
        solver="gurobi",  # Should be overridden by env var
        gap_rel=0.0,
        time_limit=None,
    )
    
    with patch("pulp.PULP_CBC_CMD") as MockCbcCmd:
        mock_solver = Mock()
        MockCbcCmd.return_value = mock_solver
        
        solver = pick_solver(params)
        
        # Should use CBC (from env var)
        MockCbcCmd.assert_called_once()


def test_gurobi_adapter_properties():
    """Test GurobiAdapter properties."""
    adapter = GurobiAdapter()
    
    assert adapter.name == "Gurobi"
    assert isinstance(adapter.available, bool)


def test_cbc_adapter_properties():
    """Test CbcAdapter properties."""
    adapter = CbcAdapter()
    
    assert adapter.name == "CBC"
    assert adapter.available is True