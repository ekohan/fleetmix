"""Unit tests for henke_heterogeneous_runner orchestration + helpers."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from fleetmix.app import app
from fleetmix.experiments.reproduce_paper.henke_heterogeneous_runner import (
    _get_experiment_plan,
    _load_results,
    _parse_supply,
    _project_paths,
    _save_results,
)

runner = CliRunner()


def test_project_paths_resolve_to_package():
    config_path, datasets_dir, default_output = _project_paths()
    assert config_path.name == "base_config.yaml"
    assert "henke_heterogeneous" in config_path.parts
    assert datasets_dir.name == "mcvrp"
    assert isinstance(default_output, Path)


def test_experiment_plan_covers_150_instances_by_2_methods():
    plan = _get_experiment_plan()
    assert len(plan) == 300  # 150 instances × {BHH, TSP}
    methods = {m for _, m in plan}
    assert methods == {"BHH", "TSP"}
    # Every instance appears once per method
    bhh_instances = [i for i, m in plan if m == "BHH"]
    tsp_instances = [i for i, m in plan if m == "TSP"]
    assert set(bhh_instances) == set(tsp_instances)
    assert len(bhh_instances) == 150


def test_parse_supply_extracts_fifth_token():
    assert _parse_supply("2015_10_3_3_2_(27)") == 2
    assert _parse_supply("2015_10_3_3_3_(03)") == 3


def test_results_csv_roundtrip(tmp_path):
    csv_path = tmp_path / "hetero_all.csv"
    payload = {
        ("2015_10_3_3_1_(01)", "BHH"): {
            "instance": "2015_10_3_3_1_(01)",
            "supply": "1",
            "method": "BHH",
            "h_vehicles": "2",
            "h_cost": "310.00",
            "h_clusters": "40",
            "h_time": "1.2",
            "h_n_A": "1",
            "h_n_B": "1",
            "e_vehicles": "2",
            "e_cost": "305.00",
            "e_clusters": "1200",
            "e_time": "4.8",
            "e_n_A": "1",
            "e_n_B": "1",
            "v_gap": "0",
            "cost_gap_pct": "1.64",
        }
    }
    _save_results(csv_path, payload)
    assert csv_path.exists()
    round_tripped = _load_results(csv_path)
    assert round_tripped == payload


def test_load_results_returns_empty_when_file_missing(tmp_path):
    assert _load_results(tmp_path / "nope.csv") == {}


def test_heterogeneous_henke_experiment_mode(tmp_path, monkeypatch):
    """Mock _run_instance; verify CLI writes CSV with limit pairs."""
    fake_row = {field: "0" for field in _row_fields()}
    fake_row["instance"] = "stub"
    fake_row["method"] = "BHH"

    call_log: list[tuple[str, str]] = []

    def fake_run(instance_name, method, config, datasets_dir):
        call_log.append((instance_name, method))
        row = dict(fake_row)
        row["instance"] = instance_name
        row["method"] = method
        return row

    monkeypatch.setattr(
        "fleetmix.experiments.reproduce_paper.henke_heterogeneous_runner._run_instance",
        fake_run,
    )

    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "heterogeneous-henke",
            "--mode",
            "experiment",
            "--output",
            str(tmp_path),
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert len(call_log) == 2
    csv_path = tmp_path / "hetero_all.csv"
    assert csv_path.exists()
    rows = _load_results(csv_path)
    assert len(rows) == 2


def test_heterogeneous_henke_experiment_skips_existing(tmp_path, monkeypatch):
    """Pre-populate CSV with one row; runner should not redo that (instance, method)."""
    csv_path = tmp_path / "hetero_all.csv"
    plan = _get_experiment_plan()
    done_inst, done_meth = plan[0]
    existing_row = {field: "0" for field in _row_fields()}
    existing_row["instance"] = done_inst
    existing_row["method"] = done_meth
    _save_results(csv_path, {(done_inst, done_meth): existing_row})

    call_log: list[tuple[str, str]] = []

    def fake_run(instance_name, method, config, datasets_dir):
        call_log.append((instance_name, method))
        row = {field: "0" for field in _row_fields()}
        row["instance"] = instance_name
        row["method"] = method
        return row

    monkeypatch.setattr(
        "fleetmix.experiments.reproduce_paper.henke_heterogeneous_runner._run_instance",
        fake_run,
    )

    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "heterogeneous-henke",
            "--mode",
            "experiment",
            "--output",
            str(tmp_path),
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # The pre-existing pair should be skipped; next planned pair is run instead.
    assert (done_inst, done_meth) not in call_log
    assert len(call_log) == 1


def test_heterogeneous_henke_tsp_of_all_mode(tmp_path, monkeypatch):
    """Mock the pipeline inside run_tsp_of_all; verify CSV written with 150 rows."""
    import pandas as pd

    from fleetmix.benchmarking.models.instance_spec import InstanceSpec
    from fleetmix.core_types import DepotLocation, VehicleSpec

    dummy_customers_df = pd.DataFrame(
        {"Customer_ID": ["C1"], "Latitude": [4.5], "Longitude": [-74.0], "Demand": [1]}
    )
    dummy_spec = InstanceSpec(
        expected_vehicles=1,
        depot=DepotLocation(latitude=4.4, longitude=-73.9),
        goods=["Dry"],
        vehicles={
            "MCVRP": VehicleSpec(
                capacity=100,
                fixed_cost=100,
                compartments={"Dry": True},
                avg_speed=30,
                service_time=25,
                max_route_time=10,
            )
        },
    )

    def fake_convert(kind, *, instance_path):
        return dummy_customers_df, dummy_spec

    monkeypatch.setattr(
        "fleetmix.benchmarking.converters.vrp.convert_vrp_to_fsm", fake_convert
    )

    def fake_estimate(cluster_customers, depot, service_time, avg_speed, method):
        return 9.5, []  # below 10h threshold

    monkeypatch.setattr("fleetmix.utils.route_time.estimate_route_time", fake_estimate)

    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "heterogeneous-henke",
            "--mode",
            "tsp-of-all",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    out_csv = tmp_path / "tsp_of_all.csv"
    assert out_csv.exists()
    # 150 Henke instances; every row below 10h → all "0" in exceeds_10h
    contents = out_csv.read_text().strip().splitlines()
    assert len(contents) == 151  # header + 150 rows


def _row_fields() -> list[str]:
    from fleetmix.experiments.reproduce_paper.henke_heterogeneous_runner import (
        CSV_FIELDS,
    )

    return list(CSV_FIELDS)
