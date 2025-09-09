import pytest
from typer.testing import CliRunner
from fleetmix.app import app

def test_exp_run_smoke(monkeypatch):
    # Mock the main function directly to avoid any heavy computation
    def mock_main(*args, **kwargs):
        print("Saved summary to mocked_path.parquet")
        return None

    # Patch the main function that gets called by the CLI
    monkeypatch.setattr("fleetmix.experiments.alpha_analysis.run_grid.main", mock_main)

    r = CliRunner().invoke(app, ["exp", "run", "-e", "alpha_analysis"])
    assert r.exit_code == 0
    assert "Saved summary" in r.stdout