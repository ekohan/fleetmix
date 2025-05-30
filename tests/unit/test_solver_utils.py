"""Test solver utility functions"""
import pytest
import os
from unittest.mock import patch, MagicMock
import pulp
from fleetmix.utils.solver import pick_solver


class TestSolverUtils:
    """Test solver utility functions"""
    
    def test_pick_solver_explicit_gurobi(self):
        """Test pick_solver returns GUROBI_CMD when FSM_SOLVER=gurobi"""
        with patch.dict(os.environ, {'FSM_SOLVER': 'gurobi'}):
            with patch('pulp.GUROBI_CMD') as mock_gurobi:
                mock_solver = MagicMock()
                mock_gurobi.return_value = mock_solver
                
                result = pick_solver(verbose=False)
                
                assert result == mock_solver
                mock_gurobi.assert_called_once_with(msg=0)
    
    def test_pick_solver_explicit_cbc(self):
        """Test pick_solver returns PULP_CBC_CMD when FSM_SOLVER=cbc"""
        with patch.dict(os.environ, {'FSM_SOLVER': 'cbc'}):
            with patch('pulp.PULP_CBC_CMD') as mock_cbc:
                mock_solver = MagicMock()
                mock_cbc.return_value = mock_solver
                
                result = pick_solver(verbose=True)
                
                assert result == mock_solver
                mock_cbc.assert_called_once_with(msg=1)
    
    def test_pick_solver_auto_fallback_to_cbc(self):
        """Test pick_solver falls back to CBC when Gurobi fails and FSM_SOLVER=auto"""
        with patch.dict(os.environ, {'FSM_SOLVER': 'auto'}):
            with patch('pulp.GUROBI_CMD') as mock_gurobi:
                with patch('pulp.PULP_CBC_CMD') as mock_cbc:
                    # Gurobi raises error
                    mock_gurobi.side_effect = pulp.PulpError("Gurobi not available")
                    mock_solver = MagicMock()
                    mock_cbc.return_value = mock_solver
                    
                    result = pick_solver(verbose=False)
                    
                    assert result == mock_solver
                    mock_gurobi.assert_called_once_with(msg=0)
                    mock_cbc.assert_called_once_with(msg=0)
    
    def test_pick_solver_auto_fallback_oserror(self):
        """Test pick_solver falls back to CBC on OSError"""
        with patch.dict(os.environ, {'FSM_SOLVER': 'auto'}):
            with patch('pulp.GUROBI_CMD') as mock_gurobi:
                with patch('pulp.PULP_CBC_CMD') as mock_cbc:
                    # Gurobi raises OSError
                    mock_gurobi.side_effect = OSError("Gurobi binary not found")
                    mock_solver = MagicMock()
                    mock_cbc.return_value = mock_solver
                    
                    result = pick_solver(verbose=True)
                    
                    assert result == mock_solver
                    mock_gurobi.assert_called_once_with(msg=1)
                    mock_cbc.assert_called_once_with(msg=1) 