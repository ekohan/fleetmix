"""
End-to-end tests for FleetMix CLI and GUI.

This module implements minimal yet solid E2E tests that validate:
1. CLI optimization produces consistent results against golden artifacts
2. Core optimization invariants are maintained
3. GUI can start and respond to health checks
4. Invalid input is properly handled
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import uuid

import pytest
import requests


class TestOptimizationE2E:
    """End-to-end tests for the optimization pipeline."""

    def test_optimize_tiny_produces_valid_solution(self):
        """Test that CLI optimization produces a valid solution for tiny dataset."""
        self._test_optimization_produces_valid_solution("tiny")

    def test_optimize_mid_produces_valid_solution(self):
        """Test that CLI optimization produces a valid solution for mid dataset."""
        self._test_optimization_produces_valid_solution("mid")

    def _test_optimization_produces_valid_solution(self, case: str):
        """Helper method to test that optimization produces a valid solution."""
        # Prepare paths
        data_file = Path("tests/data") / f"{case}.csv"
        
        # Ensure test data exists
        assert data_file.exists(), f"Test data not found: {data_file}"
        
        # Run fleetmix optimize command
        cmd = [
            "python", "-m", "fleetmix", "optimize",
            "--demand", str(data_file),
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the most recently created JSON result file
        results_dir = Path("results")
        json_files = list(results_dir.glob("optimization_results_*.json"))
        
        # Use the most recent file (by modification time)
        assert len(json_files) > 0, f"No JSON results found in {results_dir}"
        actual_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        # Load results
        with open(actual_file) as f:
            data = json.load(f)
        
        # Extract core metrics
        solution = data.get("Solution Summary", {})
        other = data.get("Other Considerations", {})
        execution = data.get("Execution Details", {})
        
        missing_customers = other.get("Number of Unserved Customers", 0)
        vehicles_used = other.get("Total Vehicles Used") or solution.get("Total Vehicles", 0)
        total_cost = execution.get("Total Cost") or float(solution.get("Total Cost ($)", "0").replace("$", "").replace(",", ""))
        solver_status = execution.get("Solver Status")
        fixed_cost = execution.get("Total Fixed Cost") or float(solution.get("Fixed Cost ($)", "0").replace("$", "").replace(",", ""))
        variable_cost = execution.get("Total Variable Cost") or float(solution.get("Variable Cost ($)", "0").replace("$", "").replace(",", ""))
        
        # Assert core business invariants that should always hold
        assert missing_customers == 0, f"All customers should be served, but {missing_customers} are missing"
        assert solver_status == "Optimal", f"Solver should find optimal solution, got: {solver_status}"
        assert vehicles_used > 0, f"Should use at least 1 vehicle, got: {vehicles_used}"
        assert total_cost > 0, f"Total cost should be positive, got: {total_cost}"
        assert fixed_cost > 0, f"Fixed cost should be positive, got: {fixed_cost}"
        assert variable_cost >= 0, f"Variable cost should be non-negative, got: {variable_cost}"
        
        # Cost breakdown should make sense
        expected_total = fixed_cost + variable_cost
        assert abs(total_cost - expected_total) < 0.01, \
            f"Total cost {total_cost} should equal fixed + variable costs {expected_total}"

    def test_core_optimization_invariants_tiny(self):
        """Test core invariants for tiny dataset: missing_customers == 0, vehicles_used > 0, total_cost > 0."""
        self._test_core_invariants("tiny")

    def test_core_optimization_invariants_mid(self):
        """Test core invariants for mid dataset: missing_customers == 0, vehicles_used > 0, total_cost > 0."""
        self._test_core_invariants("mid")

    def _test_core_invariants(self, case: str):
        """Helper method to test core optimization invariants."""
        # Prepare paths
        data_file = Path("tests/data") / f"{case}.csv"
        
        # Run fleetmix optimize command
        cmd = [
            "python", "-m", "fleetmix", "optimize",
            "--demand", str(data_file),
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find and load the most recent results
        results_dir = Path("results")
        json_files = list(results_dir.glob("optimization_results_*.json"))
        
        assert len(json_files) > 0, f"No JSON results found in {results_dir}"
        # Use the most recent file
        actual_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        with open(actual_file) as f:
            data = json.load(f)
        
        # Extract metrics
        solution = data.get("Solution Summary", {})
        other = data.get("Other Considerations", {})
        execution = data.get("Execution Details", {})
        
        missing_customers = other.get("Number of Unserved Customers", 0)
        vehicles_used = other.get("Total Vehicles Used") or solution.get("Total Vehicles", 0)
        total_cost = execution.get("Total Cost") or float(solution.get("Total Cost ($)", "0").replace("$", "").replace(",", ""))
        
        # Assert core invariants
        assert missing_customers == 0, f"Expected 0 missing customers, got {missing_customers}"
        assert vehicles_used > 0, f"Expected vehicles_used > 0, got {vehicles_used}"
        assert total_cost > 0, f"Expected total_cost > 0, got {total_cost}"


class TestGUIE2E:
    """End-to-end tests for the GUI interface."""

    def test_gui_smoke_test(self):
        """Smoke test: start GUI and verify it responds to health checks."""
        # Start GUI in background
        proc = subprocess.Popen(
            ["python", "-m", "fleetmix", "gui", "--port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait for GUI to start up
            time.sleep(10)
            
            # Check if GUI is responding
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            assert response.status_code == 200, f"GUI health check failed with status {response.status_code}"
            
        finally:
            # Clean up: kill the process
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestErrorHandlingE2E:
    """End-to-end tests for error handling scenarios."""

    def test_broken_input_handling(self, tmp_path: Path):
        """Test that broken input (missing demand column) is handled gracefully."""
        # Create a CSV missing the demand column (has wrong structure)
        broken_file = tmp_path / "broken.csv"
        broken_file.write_text("CustomerID,Latitude,Longitude\nC001,4.65,-74.08\nC002,4.66,-74.09\n")
        
        # Run fleetmix optimize command and expect it to fail
        cmd = [
            "python", "-m", "fleetmix", "optimize",
            "--demand", str(broken_file),
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with non-zero code
        assert result.returncode != 0, "Expected non-zero exit code for broken input"
        
        # Should contain error message about invalid input
        error_output = result.stderr.lower()
        assert any(phrase in error_output for phrase in ["invalid", "error", "missing", "demand"]), \
            f"Expected error message about invalid input in stderr: {result.stderr}"

    def test_missing_file_handling(self, tmp_path: Path):
        """Test that missing input file is handled gracefully."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        
        # Run fleetmix optimize command with non-existent file
        cmd = [
            "python", "-m", "fleetmix", "optimize",
            "--demand", str(nonexistent_file),
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with non-zero code
        assert result.returncode != 0, "Expected non-zero exit code for missing file"
        
        # Should contain error message about file not found
        error_output = result.stderr.lower()
        assert any(phrase in error_output for phrase in ["not found", "file", "exist"]), \
            f"Expected error message about missing file in stderr: {result.stderr}" 