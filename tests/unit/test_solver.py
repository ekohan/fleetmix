"""Test the solver module."""

import os
from unittest.mock import MagicMock, patch

from fleetmix.utils.solver import pick_solver


def test_pick_solver_default(monkeypatch):
    """Test pick_solver with default auto mode (FSM_SOLVER unset)."""
    # Temporarily remove FSM_SOLVER just for this test
    monkeypatch.delenv("FSM_SOLVER", raising=False)

    solver = pick_solver(verbose=False)
    # Should return a solver (either CBC or Gurobi)
    assert solver is not None


@patch.dict(os.environ, {"FSM_SOLVER": "cbc"})
def test_pick_solver_cbc():
    """Test pick_solver with CBC explicitly selected."""
    solver = pick_solver(verbose=False)
    # Should return CBC solver
    assert "CBC" in str(type(solver))


@patch.dict(os.environ, {"FSM_SOLVER": "gurobi"})
@patch("pulp.GUROBI_CMD")
def test_pick_solver_gurobi(mock_gurobi):
    """Test pick_solver with Gurobi explicitly selected."""
    mock_solver = MagicMock()
    mock_gurobi.return_value = mock_solver

    solver = pick_solver(verbose=False)

    # Should call GUROBI_CMD
    mock_gurobi.assert_called_once_with(msg=0, gapRel=0, options=[('TimeLimit', 180)])
    assert solver == mock_solver


@patch.dict(os.environ, {"FSM_SOLVER": "auto"})
@patch("pulp.GUROBI_CMD")
@patch("pulp.PULP_CBC_CMD")
def test_pick_solver_auto_fallback(mock_cbc, mock_gurobi):
    """Test pick_solver auto mode falls back to CBC when Gurobi fails."""
    import pulp

    # Make Gurobi fail
    mock_gurobi.side_effect = pulp.PulpError("Gurobi not available")
    mock_cbc_solver = MagicMock()
    mock_cbc.return_value = mock_cbc_solver

    solver = pick_solver(verbose=True)

    # Should try Gurobi first, then fall back to CBC
    mock_gurobi.assert_called_once_with(msg=1, gapRel=0, options=[('TimeLimit', 180)])
    mock_cbc.assert_called_once_with(msg=1, gapRel=0)
    assert solver == mock_cbc_solver


def test_pick_solver_verbose(monkeypatch):
    """Test pick_solver with verbose mode when FSM_SOLVER is unset."""
    monkeypatch.delenv("FSM_SOLVER", raising=False)

    # Just check it doesn't crash with verbose=True
    solver = pick_solver(verbose=True)
    assert solver is not None
