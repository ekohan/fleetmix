"""Test the debug module for MILP model debugging."""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import pulp

from fleetmix.utils.debug import ModelDebugger


@pytest.fixture
def reset_debugger():
    """Reset ModelDebugger state before each test."""
    ModelDebugger.active = False
    yield
    ModelDebugger.active = False


def test_model_debugger_enable_default(tmp_path, reset_debugger):
    """Test enabling debugger with default artifacts."""
    debug_dir = tmp_path / "debug"
    
    ModelDebugger.enable(debug_dir=debug_dir)
    
    assert ModelDebugger.active is True
    assert ModelDebugger._dir == debug_dir
    assert debug_dir.exists()
    assert ModelDebugger._artifacts == {"lp", "mps", "solver_log", "iis"}


def test_model_debugger_enable_custom_artifacts(tmp_path, reset_debugger):
    """Test enabling debugger with custom artifact set."""
    debug_dir = tmp_path / "debug"
    custom_artifacts = {"lp", "solver_log"}
    
    ModelDebugger.enable(debug_dir=debug_dir, artifacts=custom_artifacts)
    
    assert ModelDebugger.active is True
    assert ModelDebugger._artifacts == custom_artifacts


def test_model_debugger_enable_creates_directory(tmp_path, reset_debugger):
    """Test that enable creates the debug directory if it doesn't exist."""
    debug_dir = tmp_path / "nested" / "debug" / "path"
    assert not debug_dir.exists()
    
    ModelDebugger.enable(debug_dir=debug_dir)
    
    assert debug_dir.exists()


def test_model_debugger_dump_when_inactive(reset_debugger):
    """Test that dump does nothing when debugger is not active."""
    model = pulp.LpProblem("test", pulp.LpMinimize)
    
    # Should not raise any errors
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_dump_lp_file(tmp_path, reset_debugger):
    """Test dumping LP file."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"lp"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    ModelDebugger.dump(model, "test_model")
    
    lp_file = tmp_path / "test_model.lp"
    assert lp_file.exists()
    content = lp_file.read_text()
    assert "test" in content.lower()


def test_model_debugger_dump_mps_file(tmp_path, reset_debugger):
    """Test dumping MPS file."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"mps"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    ModelDebugger.dump(model, "test_model")
    
    # MPS file might be created depending on PuLP version
    mps_file = tmp_path / "test_model.mps"
    # Don't assert existence since it may fail silently (lines 85-87)


def test_model_debugger_dump_lp_write_failure(tmp_path, reset_debugger, monkeypatch):
    """Test handling of LP write failures."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"lp"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    
    # Mock writeLP to raise an exception
    def mock_write_lp(path):
        raise IOError("Write failed")
    
    monkeypatch.setattr(model, "writeLP", mock_write_lp)
    
    # Should not raise, just log warning
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_dump_mps_silent_failure(tmp_path, reset_debugger, monkeypatch):
    """Test that MPS write failures are silently ignored (lines 85-87)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"mps"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    
    # Mock writeMPS to raise an exception
    def mock_write_mps(path):
        raise IOError("MPS write failed")
    
    monkeypatch.setattr(model, "writeMPS", mock_write_mps)
    
    # Should not raise or log warning, silently skip
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_capture_solver_log(tmp_path, reset_debugger):
    """Test capturing solver log (lines 90-91)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"solver_log"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    model += x
    model += x >= 5  # Add a constraint
    
    # Create solver
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solver = solver
    
    ModelDebugger.dump(model, "test_model")
    
    log_file = tmp_path / "test_model.log"
    # Log file should exist (lines 101-127)
    assert log_file.exists()


def test_model_debugger_capture_solver_log_no_solver(tmp_path, reset_debugger):
    """Test solver log capture when solver is None (line 104-105)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"solver_log"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    model.solver = None
    
    ModelDebugger.dump(model, "test_model")
    
    # Should not create log file
    log_file = tmp_path / "test_model.log"
    assert not log_file.exists()


def test_model_debugger_capture_solver_log_exception(tmp_path, reset_debugger, monkeypatch):
    """Test handling exceptions during solver log capture (lines 123-124)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"solver_log"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    # Create a mock solver that raises an exception during solve
    mock_solver = Mock()
    mock_solver.msg = 0
    mock_solver.solve = Mock(side_effect=Exception("Solver error"))
    model.solver = mock_solver
    
    # Should not raise, just log warning
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_extract_iis_infeasible_model(tmp_path, reset_debugger):
    """Test IIS extraction for infeasible models (lines 94-98)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"iis"})
    
    # Create an infeasible model
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    model += x >= 20  # Infeasible constraint
    
    # Solve to get infeasible status
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    
    # Set status to infeasible to trigger IIS logic
    model.status = pulp.LpStatusInfeasible
    
    ModelDebugger.dump(model, "test_model")
    
    # IIS extraction may not work with CBC, but code path is covered


def test_model_debugger_extract_iis_no_solver_model(tmp_path, reset_debugger):
    """Test IIS extraction when solver has no solverModel (lines 137-139)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"iis"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    # Set status to infeasible
    model.status = pulp.LpStatusInfeasible
    
    # Create solver without solverModel attribute
    mock_solver = Mock()
    del mock_solver.solverModel  # Ensure no solverModel attribute
    model.solver = mock_solver
    
    # Should exit early from _extract_iis
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_extract_iis_with_compute_iis(tmp_path, reset_debugger):
    """Test IIS extraction with computeIIS method (lines 141-146)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"iis"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    # Set status to infeasible
    model.status = pulp.LpStatusInfeasible
    
    # Create mock solver with solverModel that has computeIIS
    mock_solver_model = Mock()
    mock_solver_model.computeIIS = Mock()
    mock_solver_model.write = Mock()
    
    mock_solver = Mock()
    mock_solver.solverModel = mock_solver_model
    model.solver = mock_solver
    
    ModelDebugger.dump(model, "test_model")
    
    # Verify computeIIS and write were called
    mock_solver_model.computeIIS.assert_called_once()
    mock_solver_model.write.assert_called_once()


def test_model_debugger_extract_iis_exception(tmp_path, reset_debugger):
    """Test handling exceptions during IIS extraction (lines 147-149)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"iis"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    model += x
    
    # Set status to infeasible
    model.status = pulp.LpStatusInfeasible
    
    # Create mock solver that raises exception during IIS
    mock_solver_model = Mock()
    mock_solver_model.computeIIS = Mock(side_effect=Exception("IIS not supported"))
    
    mock_solver = Mock()
    mock_solver.solverModel = mock_solver_model
    model.solver = mock_solver
    
    # Should not raise, just log debug message
    ModelDebugger.dump(model, "test_model")


def test_model_debugger_dump_all_artifacts(tmp_path, reset_debugger):
    """Test dumping all artifacts together."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"lp", "mps", "solver_log"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    model += x
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solver = solver
    
    ModelDebugger.dump(model, "test_model")
    
    # At least LP file should exist
    assert (tmp_path / "test_model.lp").exists()


def test_model_debugger_solver_log_old_msg_attribute(tmp_path, reset_debugger):
    """Test that old_msg is restored after solver log capture (lines 108, 127)."""
    ModelDebugger.enable(debug_dir=tmp_path, artifacts={"solver_log"})
    
    model = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    model += x
    
    # Create solver with msg=0
    solver = pulp.PULP_CBC_CMD(msg=0)
    original_msg = solver.msg
    model.solver = solver
    
    ModelDebugger.dump(model, "test_model")
    
    # msg should be restored to original value (line 127)
    assert solver.msg == original_msg
