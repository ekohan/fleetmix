"""Unit tests for helper utilities in `fleetmix.app`."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import typer

from fleetmix.app import (
    _find_config_by_id,
    _get_available_instances,
    _list_instances,
    _run_all_case_instances,
    _run_all_mcvrp_instances,
    _run_single_instance,
    _setup_logging_from_flags,
)
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams
from fleetmix.core_types import VehicleConfiguration


@pytest.fixture(scope="module")
def default_params() -> FleetmixParams:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fleetmix"
        / "config"
        / "default_config.yaml"
    )
    return load_fleetmix_params(config_path)


def make_params(base: FleetmixParams, tmp_path: Path, **problem_overrides) -> FleetmixParams:
    problem = dataclasses.replace(base.problem, **problem_overrides)
    return dataclasses.replace(
        base,
        problem=problem,
        io=dataclasses.replace(base.io, results_dir=tmp_path),
    )


def test_find_config_by_id_success() -> None:
    configs = [
        VehicleConfiguration(
            config_id="1",
            vehicle_type="Truck",
            capacity=100,
            fixed_cost=100.0,
            compartments={"Dry": True},
        ),
        VehicleConfiguration(
            config_id="2",
            vehicle_type="Van",
            capacity=80,
            fixed_cost=75.0,
            compartments={"Dry": True},
        ),
    ]
    result = _find_config_by_id(configs, "2")
    assert result.vehicle_type == "Van"


def test_find_config_by_id_missing() -> None:
    with pytest.raises(KeyError):
        _find_config_by_id([], "missing")


def test_get_available_instances_mcvrp(tmp_path, monkeypatch):
    app_dir = tmp_path / "app_loc"
    dataset_dir = app_dir / "benchmarking" / "datasets" / "mcvrp"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "alpha.dat").write_text("data")
    (dataset_dir / "beta.dat").write_text("data")

    monkeypatch.setattr("fleetmix.app.__file__", str(app_dir / "app.py"), raising=False)

    assert _get_available_instances("mcvrp") == ["alpha", "beta"]


def test_get_available_instances_cvrp(tmp_path, monkeypatch):
    app_dir = tmp_path / "app_loc"
    dataset_dir = app_dir / "benchmarking" / "datasets" / "cvrp"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "X-n10-k3.vrp").write_text("data")

    monkeypatch.setattr("fleetmix.app.__file__", str(app_dir / "app.py"), raising=False)

    assert _get_available_instances("cvrp") == ["X-n10-k3"]


def test_get_available_instances_invalid_suite():
    assert _get_available_instances("invalid") == []


@patch("fleetmix.app.console.print")
def test_list_instances_empty(mock_print):
    _list_instances("unknown")
    mock_print.assert_called_with("[yellow]No instances found for UNKNOWN[/yellow]")


@patch("fleetmix.app.setup_logging")
@pytest.mark.parametrize(
    "verbose,quiet,debug,expected",
    [
        (False, False, False, None),
        (True, False, False, "VERBOSE"),
        (False, True, False, "QUIET"),
        (False, False, True, "DEBUG"),
    ],
)
def test_setup_logging_from_flags(mock_setup_logging, verbose, quiet, debug, expected):
    _setup_logging_from_flags(verbose=verbose, quiet=quiet, debug=debug)
    if expected is None:
        mock_setup_logging.assert_called_once_with()
    else:
        from fleetmix.utils.logging import LogLevel

        mock_setup_logging.assert_called_once_with(getattr(LogLevel, expected))


@patch("fleetmix.app.setup_logging")
def test_setup_logging_priority(mock_setup_logging):
    from fleetmix.utils.logging import LogLevel

    _setup_logging_from_flags(verbose=True, quiet=True, debug=True)
    mock_setup_logging.assert_called_with(LogLevel.DEBUG)

    _setup_logging_from_flags(verbose=True, quiet=True, debug=False)
    mock_setup_logging.assert_called_with(LogLevel.VERBOSE)


@pytest.mark.parametrize(
    "suite,instance,expected_file",
    [
        ("mcvrp", "ut_temp_mcvrp", "mcvrp_ut_temp_mcvrp.json"),
        ("cvrp", "ut_temp_cvrp", "cvrp_ut_temp_cvrp_normal.json"),
        ("case", "ut_temp_case", "case_ut_temp_case.json"),
    ],
)
@patch("fleetmix.app.save_optimization_results")
@patch("fleetmix.app.api.optimize")
@patch("fleetmix.app.convert_to_fsm")
@patch("fleetmix.app.FleetmixParams.apply_instance_spec", lambda self, spec: self)
@patch("fleetmix.app.log_success")
@patch("fleetmix.app.log_progress")
def test_run_single_instance_success(
    _mock_progress,
    _mock_success,
    mock_convert,
    mock_run,
    mock_save,
    suite,
    instance,
    expected_file,
    tmp_path,
    default_params,
    monkeypatch,
):
    monkeypatch.setenv("FLEETMIX_SKIP_OPTIMISE", "0")
    mock_convert.return_value = (pd.DataFrame(), object())
    mock_run.return_value = MagicMock(
        total_cost=1.0,
        total_fixed_cost=0.5,
        total_variable_cost=0.5,
        total_penalties=0.0,
        vehicles_used=1,
        solver_status="ok",
        solver_runtime_sec=0.1,
        missing_customers=[],
        selected_clusters=[],
    )

    datasets_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
    )
    created_file: Path | None = None

    if suite == "mcvrp":
        created_file = datasets_root / "mcvrp" / f"{instance}.dat"
        created_file.write_text("data")
    elif suite == "cvrp":
        monkeypatch.setattr(
            "fleetmix.app._get_available_instances",
            lambda suite_name: [instance] if suite_name == "cvrp" else [],
        )
    elif suite == "case":
        created_file = datasets_root / "case" / f"{instance}.csv"
        created_file.write_text(
            "Customer_ID,Latitude,Longitude,Demand_Type,Units_Demand\n"
            "C1,0,0,dry,1\n"
        )
        monkeypatch.setattr(
            "fleetmix.utils.data_processing.load_customer_demand",
            lambda _: pd.DataFrame(
                {
                    "Num_Customers": [1],
                    "Customer_ID": ["C1"],
                    "Latitude": [0],
                    "Longitude": [0],
                    "Dry_Demand": [1],
                }
            ),
        )

    params = make_params(default_params, tmp_path)
    monkeypatch.setattr("fleetmix.app.load_fleetmix_params", lambda path: params)

    try:
        _run_single_instance(suite, instance, tmp_path, "json", False, None, None)
    finally:
        if created_file and created_file.exists():
            created_file.unlink()

    mock_save.assert_called_once()
    saved_path = mock_save.call_args.kwargs["filename"]
    assert saved_path.endswith(expected_file)


@pytest.mark.parametrize(
    "suite,missing_message",
    [
        ("mcvrp", "MCVRP instance"),
        ("cvrp", "CVRP instance"),
        ("case", "Case instance"),
    ],
)
def test_run_single_instance_missing_file(suite, missing_message, tmp_path):
    with pytest.raises(typer.Exit):
        _run_single_instance(suite, "missing", tmp_path, "json", False, None, None)


@patch("fleetmix.app.log_error")
@patch("fleetmix.app.api.optimize")
@patch("fleetmix.app.convert_to_fsm", return_value=(pd.DataFrame(), MagicMock()))
@patch("fleetmix.app.load_fleetmix_params")
def test_run_all_mcvrp_instances_handles_errors(
    mock_load,
    mock_convert,
    mock_run,
    mock_log_error,
    tmp_path,
    default_params,
    monkeypatch,
):
    mock_run.side_effect = [MagicMock(), RuntimeError("boom")]
    mock_load.return_value = default_params

    datasets_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "mcvrp"
    )
    created_files = [datasets_dir / "unit_inst_ok.dat", datasets_dir / "unit_inst_fail.dat"]
    for path in created_files:
        path.write_text("dummy data\n")

    original_glob = Path.glob

    def fake_glob(self, pattern):
        if self == datasets_dir and pattern == "*.dat":
            return created_files
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", fake_glob)

    try:
        _run_all_mcvrp_instances(tmp_path)
    finally:
        for path in created_files:
            path.unlink(missing_ok=True)

    assert mock_run.call_count == 2
    assert mock_log_error.call_count >= 1
    error_messages = " ".join(call.args[0] for call in mock_log_error.call_args_list)
    assert "unit_inst_fail" in error_messages


def test_run_all_case_instances_success(
    tmp_path: Path, default_params: FleetmixParams, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "case"
    )
    sample_file = sorted(datasets_dir.glob("*.csv"))[0]

    monkeypatch.setattr(
        "fleetmix.app.Path.glob",
        lambda self, pattern: [sample_file] if self == datasets_dir and pattern == "*.csv" else [],
    )
    monkeypatch.setattr("fleetmix.app.load_fleetmix_params", lambda path: default_params)
    monkeypatch.setattr("fleetmix.app._DEFAULT_CONFIG", default_params)

    monkeypatch.setattr(
        "fleetmix.utils.data_processing.load_customer_demand",
        lambda _: pd.DataFrame(
            {
                "Num_Customers": [1],
                "Customer_ID": ["C1"],
                "Latitude": [0],
                "Longitude": [0],
                "Dry_Demand": [1],
            }
        ),
    )

    run_calls: list[Path] = []
    def fake_run(**kwargs):
        # api.optimize signature: demand, config, output_dir, format, verbose
        params = kwargs.get("config")
        if params:
            run_calls.append(params.io.results_dir)
        return MagicMock(
            total_cost=1.0,
            total_fixed_cost=0.5,
            total_variable_cost=0.5,
            total_penalties=0.0,
            total_vehicles=1,
            missing_customers=[],
            solver_status="ok",
            solver_runtime_sec=0.1,
        )

    saves: list[str] = []

    monkeypatch.setattr("fleetmix.app.api.optimize", fake_run)
    monkeypatch.setattr(
        "fleetmix.app.save_optimization_results",
        lambda **kwargs: saves.append(kwargs["filename"]),
    )

    logged_errors: list[str] = []
    monkeypatch.setattr("fleetmix.app.log_error", lambda msg: logged_errors.append(msg))

    _run_all_case_instances(tmp_path)

    assert len(run_calls) == 1
    assert run_calls[0] == tmp_path
    assert len(saves) == 1
    assert saves[0].startswith(str(tmp_path))
    assert not logged_errors
