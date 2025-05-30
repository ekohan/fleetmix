"""Unit tests for the solver module."""

import unittest
import os
from unittest.mock import patch, MagicMock

from fleetmix.utils.solver import pick_solver


class TestPickSolver(unittest.TestCase):
    """Test cases for pick_solver function."""
    
    def setUp(self):
        """Clear environment variable before each test."""
        if 'FSM_SOLVER' in os.environ:
            del os.environ['FSM_SOLVER']
    
    @patch('pulp.GUROBI_CMD')
    def test_pick_solver_explicit_gurobi(self, mock_gurobi):
        """Test explicitly selecting Gurobi solver."""
        os.environ['FSM_SOLVER'] = 'gurobi'
        mock_solver = MagicMock()
        mock_gurobi.return_value = mock_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_solver)
    
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_explicit_cbc(self, mock_cbc):
        """Test explicitly selecting CBC solver."""
        os.environ['FSM_SOLVER'] = 'cbc'
        mock_solver = MagicMock()
        mock_cbc.return_value = mock_solver
        
        result = pick_solver(verbose=False)
        
        mock_cbc.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_solver)
    
    @patch('pulp.GUROBI_CMD')
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_auto_gurobi_success(self, mock_cbc, mock_gurobi):
        """Test auto mode successfully using Gurobi."""
        os.environ['FSM_SOLVER'] = 'auto'
        mock_gurobi_solver = MagicMock()
        mock_gurobi.return_value = mock_gurobi_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        mock_cbc.assert_not_called()
        self.assertEqual(result, mock_gurobi_solver)
    
    @patch('pulp.GUROBI_CMD')
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_auto_fallback_to_cbc(self, mock_cbc, mock_gurobi):
        """Test auto mode falling back to CBC when Gurobi fails."""
        os.environ['FSM_SOLVER'] = 'auto'
        # Make Gurobi fail
        mock_gurobi.side_effect = OSError("Gurobi not found")
        mock_cbc_solver = MagicMock()
        mock_cbc.return_value = mock_cbc_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        mock_cbc.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_cbc_solver)
    
    @patch('pulp.GUROBI_CMD')
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_default_auto(self, mock_cbc, mock_gurobi):
        """Test default behavior (auto mode) when no env var is set."""
        # No environment variable set
        mock_gurobi_solver = MagicMock()
        mock_gurobi.return_value = mock_gurobi_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_gurobi_solver)
    
    @patch('pulp.GUROBI_CMD')
    def test_pick_solver_verbose_mode(self, mock_gurobi):
        """Test verbose mode passes correct message level."""
        os.environ['FSM_SOLVER'] = 'gurobi'
        mock_solver = MagicMock()
        mock_gurobi.return_value = mock_solver
        
        result = pick_solver(verbose=True)
        
        mock_gurobi.assert_called_once_with(msg=1)
        self.assertEqual(result, mock_solver)
    
    @patch('pulp.GUROBI_CMD')
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_case_insensitive(self, mock_cbc, mock_gurobi):
        """Test that solver selection is case-insensitive."""
        os.environ['FSM_SOLVER'] = 'GUROBI'
        mock_solver = MagicMock()
        mock_gurobi.return_value = mock_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_solver)
    
    @patch('pulp.GUROBI_CMD')
    @patch('pulp.PULP_CBC_CMD')
    def test_pick_solver_pulp_error_fallback(self, mock_cbc, mock_gurobi):
        """Test fallback when Gurobi raises PulpError."""
        import pulp
        
        # Make Gurobi fail with PulpError
        mock_gurobi.side_effect = pulp.PulpError("Gurobi license issue")
        mock_cbc_solver = MagicMock()
        mock_cbc.return_value = mock_cbc_solver
        
        result = pick_solver(verbose=False)
        
        mock_gurobi.assert_called_once_with(msg=0)
        mock_cbc.assert_called_once_with(msg=0)
        self.assertEqual(result, mock_cbc_solver)


if __name__ == "__main__":
    unittest.main() 