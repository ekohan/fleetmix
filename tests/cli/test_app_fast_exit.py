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
MCVRP_INSTANCE = "2015_10_3_3_1_(09)"
CVRP_INSTANCE = "X-n129-k18"
CASE_INSTANCE = "sales_2024-06-01_demand"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def test_benchmark_cvrp_fast_exit_creates_placeholder(
    runner: CliRunner, base_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fleetmix.app._get_available_instances",
        lambda suite: [CVRP_INSTANCE] if suite == "cvrp" else [],
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            "cvrp",
            "--instance",
            CVRP_INSTANCE,
            "--output",
            str(tmp_path),
        ],
        env=base_env,
    )

    assert result.exit_code == 0
    placeholder = tmp_path / f"cvrp_{CVRP_INSTANCE}_normal.json"
    assert placeholder.exists()
    assert placeholder.read_text() == "{}"


def test_benchmark_mcvrp_fast_exit_creates_placeholder(
    runner: CliRunner, base_env: dict[str, str], tmp_path: Path
) -> None:
    datasets_dir = (
        _project_root()
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "mcvrp"
    )
    datasets_dir.mkdir(parents=True, exist_ok=True)
    temp_file = datasets_dir / f"{MCVRP_INSTANCE}.dat"
    temp_file.write_text("data")

    try:
        result = runner.invoke(
            app,
            [
                "benchmark",
                "mcvrp",
                "--instance",
                MCVRP_INSTANCE,
                "--output",
                str(tmp_path),
            ],
            env=base_env,
        )
    finally:
        temp_file.unlink(missing_ok=True)

    assert result.exit_code == 0
    placeholder = tmp_path / f"mcvrp_{MCVRP_INSTANCE}.json"
    assert placeholder.exists()
    assert placeholder.read_text() == "{}"


def test_benchmark_case_fast_exit_creates_placeholder(
    runner: CliRunner, base_env: dict[str, str], tmp_path: Path
) -> None:
    datasets_dir = (
        _project_root()
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "case"
    )
    datasets_dir.mkdir(parents=True, exist_ok=True)
    temp_file = datasets_dir / f"{CASE_INSTANCE}.csv"
    temp_file.write_text("data")

    try:
        result = runner.invoke(
            app,
            [
                "benchmark",
                "case",
                "--instance",
                CASE_INSTANCE,
                "--output",
                str(tmp_path),
            ],
            env=base_env,
        )
    finally:
        temp_file.unlink(missing_ok=True)

    assert result.exit_code == 0
    placeholder = tmp_path / f"case_{CASE_INSTANCE}.json"
    assert placeholder.exists()
    assert placeholder.read_text() == "{}"


def test_convert_cvrp_fast_exit_creates_placeholder(
    runner: CliRunner, base_env: dict[str, str], tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "convert",
            "--type",
            "cvrp",
            "--instance",
            CVRP_INSTANCE,
            "--benchmark-type",
            "normal",
            "--output",
            str(tmp_path),
        ],
        env=base_env,
    )

    assert result.exit_code == 0
    placeholder = tmp_path / f"vrp_cvrp_{CVRP_INSTANCE}_normal.json"
    assert placeholder.exists()
    assert placeholder.read_text() == "{}"


def test_convert_invalid_type_errors(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "convert",
            "--type",
            "invalid",
            "--instance",
            "anything",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid type" in result.stderr


def test_version_command_outputs_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_experiments_run_alpha_analysis_invokes_main(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, bool] = {}

    def fake_main(config_path: Path | None) -> None:
        called["run"] = True
        assert config_path is None

    monkeypatch.setattr(
        "fleetmix.experiments.alpha_analysis.run_grid.main", fake_main
    )

    result = runner.invoke(app, ["exp", "run", "--experiment", "alpha_analysis"])

    assert result.exit_code == 0
    assert called.get("run") is True


def test_experiments_missing_experiment_errors(runner: CliRunner, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("ERROR"):
        result = runner.invoke(app, ["exp", "run"])

    assert result.exit_code != 0
    assert "Missing --experiment" in caplog.text


def test_experiments_unknown_action_errors(runner: CliRunner, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("ERROR"):
        result = runner.invoke(app, ["exp", "unknown", "--experiment", "alpha_analysis"])

    assert result.exit_code != 0
    assert "Unknown action" in caplog.text


def test_optimize_normal_flow_creates_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demand = tmp_path / "demand.csv"
    demand.write_text(
        "Customer_ID,Latitude,Longitude,Dry_Demand\n"
        "1,0.0,0.0,5\n"
    )
    output_dir = tmp_path / "results"

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "cli-normal")
    monkeypatch.setenv("FLEETMIX_SKIP_OPTIMISE", "0")

    def fake_api_optimize(**kwargs):
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
        )

    monkeypatch.setattr("fleetmix.app.api_optimize", fake_api_optimize)

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


def test_benchmark_cvrp_normal_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "cli-normal")
    monkeypatch.setenv("FLEETMIX_SKIP_OPTIMISE", "0")

    captured: dict[str, tuple] = {}

    def fake_run(suite, instance, output, format, verbose, allow_split_stops, config):
        captured["call"] = (suite, instance, output, format, verbose, allow_split_stops, config)

    monkeypatch.setattr("fleetmix.app._run_single_instance", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "cvrp",
            "--instance",
            CVRP_INSTANCE,
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["call"] == (
        "cvrp",
        CVRP_INSTANCE,
        tmp_path,
        "json",
        False,
        None,
        None,
    )


def test_convert_cvrp_normal_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "cli-normal")
    monkeypatch.setenv("FLEETMIX_SKIP_OPTIMISE", "0")

    def fake_convert(*args, **kwargs):
        return pd.DataFrame(), SimpleNamespace(apply=lambda params: params)

    def fake_run_optimization(**kwargs):
        return SimpleNamespace(
            total_cost=1.0,
            total_fixed_cost=0.5,
            total_variable_cost=0.5,
            total_penalties=0.0,
            vehicles_used=1,
            missing_customers=[],
            solver_status="ok",
            solver_runtime_sec=0.1,
            selected_clusters=[],
        )

    saved: dict[str, str] = {}

    monkeypatch.setattr("fleetmix.app.convert_to_fsm", fake_convert)
    monkeypatch.setattr("fleetmix.app.run_optimization", lambda **kwargs: fake_run_optimization())
    monkeypatch.setattr("fleetmix.app.save_optimization_results", lambda **kwargs: saved.setdefault("filename", kwargs["filename"]))
    monkeypatch.setattr("fleetmix.app.FleetmixParams.apply_instance_spec", lambda self, spec: self)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "convert",
            "--type",
            "cvrp",
            "--instance",
            CVRP_INSTANCE,
            "--benchmark-type",
            "normal",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "filename" in saved
    assert saved["filename"].startswith(str(tmp_path))

