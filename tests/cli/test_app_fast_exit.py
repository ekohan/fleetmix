"""Additional CLI tests for `fleetmix.app` covering fast-exit and validation paths."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from fleetmix import __version__
from fleetmix.app import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def demand_file(tmp_path: Path) -> Path:
    path = tmp_path / "demand.csv"
    path.write_text(
        "Customer_ID,Latitude,Longitude,Dry_Demand\n"
        "1,0.0,0.0,5\n"
    )
    return path


@pytest.fixture
def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTEST_CURRENT_TEST"] = "cli-fast-path"
    env["FLEETMIX_SKIP_OPTIMISE"] = "1"
    return env


MINIMAL_CONFIG = Path("tests/_assets/configs/test_config_minimal.yaml")
MCVRP_INSTANCE = "2015_10_3_3_1_(00)_dummy"
CASE_INSTANCE = "sales_2024-06-01_demand"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_optimize_fast_exit_creates_output_dir(
    runner: CliRunner, demand_file: Path, base_env: dict[str, str], tmp_path: Path
) -> None:
    output_dir = tmp_path / "optimize-out"

    result = runner.invoke(
        app,
        [
            "optimize",
            "--demand",
            str(demand_file),
            "--config",
            str(MINIMAL_CONFIG),
            "--output",
            str(output_dir),
            "--format",
            "json",
        ],
        env=base_env,
    )

    assert result.exit_code == 0
    assert output_dir.exists()


def test_optimize_invalid_format_errors(
    runner: CliRunner, demand_file: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "optimize-invalid"

    result = runner.invoke(
        app,
        [
            "optimize",
            "--demand",
            str(demand_file),
            "--config",
            str(MINIMAL_CONFIG),
            "--output",
            str(output_dir),
            "--format",
            "yaml",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid format" in result.stderr






def test_version_command_outputs_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout



def test_optimize_normal_flow_creates_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demand = tmp_path / "demand.csv"
    demand.write_text(
        "Customer_ID,Latitude,Longitude,Dry_Demand,Chilled_Demand,Frozen_Demand\n"
        "1,0.0,0.0,5,0,0\n"
    )
    output_dir = tmp_path / "results"

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "cli-normal")
    monkeypatch.setenv("FLEETMIX_SKIP_OPTIMISE", "0")

    def fake_optimize(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text("{}")
        return SimpleNamespace(
            total_cost=1.0,
            total_fixed_cost=0.5,
            total_variable_cost=0.5,
            total_penalties=0.0,
            total_vehicles=1,
            missing_customers=[],
            solver_status="ok",
            solver_runtime_sec=0.1,
            time_measurements=[],
        )

    monkeypatch.setattr("fleetmix.api.optimize", fake_optimize)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
            "--demand",
            str(demand),
            "--config",
            str(MINIMAL_CONFIG),
            "--output",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "summary.json").exists()




