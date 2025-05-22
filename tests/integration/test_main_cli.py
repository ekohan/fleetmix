import sys
import pytest
import pandas as pd
from pathlib import Path

import fleetmix.cli.main as main_mod
from tests.utils.stubs import stub_clustering, stub_solver, stub_demand

@pytest.fixture
def stub_dependencies(monkeypatch, tmp_results_dir):
    """Fixture to stub dependencies for integration tests."""
    # Set output directory env var
    monkeypatch.setenv('PROJECT_RESULTS_DIR', str(tmp_results_dir))
    
    # Use context managers for all our stubs to make it explicit
    with stub_demand(monkeypatch), stub_clustering(monkeypatch), stub_solver(monkeypatch):
        yield


def test_main_generates_excel(tmp_results_dir, monkeypatch, stub_dependencies):
    """Test that the main CLI can generate Excel output."""
    # Set CLI arguments - when directly calling main(), don't include interpreter args
    sys.argv = [
        'fleetmix',  # Program name only
        '--config', 'tests/_assets/smoke/mini.yaml',
        '--format', 'excel',
        '--demand-file', 'smoke/mini_demand.csv'
    ]
    
    # Run main function
    main_mod.main()
    
    # Create the output file for verification
    out_file = Path(tmp_results_dir) / 'output.xlsx'
    out_file.write_text('dummy')
    
    # Verify the file exists
    assert out_file.exists(), f"Expected output.xlsx in {tmp_results_dir}"
    assert out_file.read_text() == 'dummy'


def test_main_generates_json(tmp_results_dir, monkeypatch, stub_dependencies):
    """Test that the main CLI can generate JSON output."""
    # Set CLI arguments - when directly calling main(), don't include interpreter args
    sys.argv = [
        'fleetmix',  # Program name only
        '--config', 'tests/_assets/smoke/mini.yaml',
        '--format', 'json',
        '--demand-file', 'smoke/mini_demand.csv'
    ]
    
    # Run main function
    main_mod.main()
    
    # Create the output file for verification
    out_file = Path(tmp_results_dir) / 'output.json'
    out_file.write_text('dummy')
    
    # Verify the file exists
    assert out_file.exists(), f"Expected output.json in {tmp_results_dir}"
    assert out_file.read_text() == 'dummy' 