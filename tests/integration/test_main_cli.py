"""Test integration of the main CLI workflow."""

from typer.testing import CliRunner

from fleetmix.app import app
from tests.utils.stubs import (
    stub_clustering,
    stub_demand,
    stub_save_results,
    stub_solver,
)

runner = CliRunner()


def test_main_generates_excel(tmp_path, monkeypatch):
    """Test that the main optimize command succeeds with Excel format."""
    # Create a dummy demand file since validation happens before stub
    demand_file = tmp_path / "dummy_demand.csv"
    demand_file.write_text(
        "Customer_ID,Latitude,Longitude,Dry_Demand,Chilled_Demand,Frozen_Demand\nC1,0,0,10,0,0\n"
    )

    with (
        stub_clustering(monkeypatch),
        stub_solver(monkeypatch),
        stub_demand(monkeypatch),
        stub_save_results(monkeypatch, tmp_path),
    ):
        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(demand_file),
                "--output",
                str(tmp_path),
                "--format",
                "xlsx",
            ],
        )

        if result.exit_code != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"Exception: {result.exception}")

        # Just verify the command succeeds - output file creation is handled by stubs
        assert result.exit_code == 0


def test_main_generates_json(tmp_path, monkeypatch):
    """Test that the main optimize command succeeds with JSON format."""
    # Create a dummy demand file since validation happens before stub
    demand_file = tmp_path / "dummy_demand.csv"
    demand_file.write_text(
        "Customer_ID,Latitude,Longitude,Dry_Demand,Chilled_Demand,Frozen_Demand\nC1,0,0,10,0,0\n"
    )

    with (
        stub_clustering(monkeypatch),
        stub_solver(monkeypatch),
        stub_demand(monkeypatch),
        stub_save_results(monkeypatch, tmp_path),
    ):
        result = runner.invoke(
            app,
            [
                "optimize",
                "--demand",
                str(demand_file),
                "--output",
                str(tmp_path),
                "--format",
                "json",
            ],
        )

        if result.exit_code != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"Exception: {result.exception}")

        # Just verify the command succeeds - output file creation is handled by stubs
        assert result.exit_code == 0
