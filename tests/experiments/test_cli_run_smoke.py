import pytest
from typer.testing import CliRunner
from fleetmix.app import app
from pathlib import Path

@pytest.mark.slow
def test_exp_run_smoke(monkeypatch):
    # Reduce the grid size to a single point for a fast smoke test
    monkeypatch.setattr("experiments.alpha_analysis.run_grid.DEMAND_FILES", [Path("mock.csv")])
    monkeypatch.setattr("experiments.alpha_analysis.run_grid.ALPHA_GRID", [1.0])
    monkeypatch.setattr("experiments.alpha_analysis.run_grid.C_VALUES", [0.0])

    # Mock the actual run_day function to avoid heavy computation
    def mock_run_day(*args, **kwargs):
        return {"status": "success"}

    monkeypatch.setattr("experiments.alpha_analysis.run_grid.run_day", mock_run_day)

    r = CliRunner().invoke(app, ["exp", "run", "-e", "alpha_analysis"])
    assert r.exit_code == 0
    assert "Saved summary" in r.stdout