"""Integration tests for 'fleetmix reproduce-paper' commands."""

import shutil
from pathlib import Path
from typer.testing import CliRunner
from fleetmix.app import app
import pytest

runner = CliRunner()

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

@pytest.fixture
def clean_results(tmp_path):
    """Fixture to provide a clean results directory."""
    output_dir = tmp_path / "repro_results"
    output_dir.mkdir()
    return output_dir

def test_reproduce_mcvrp_instances_smoke(clean_results):
    """Smoke test for 'reproduce-paper mcvrp-instances'."""
    instance_name = "2015_10_3_3_3_(14)"
    
    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "mcvrp-instances",
            "--instances",
            instance_name,
            "--output",
            str(clean_results),
            "--skip-existing", # Should run since dir is new
        ],
    )
    
    assert result.exit_code == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    # Verify output file creation
    expected_file = clean_results / f"mcvrp_{instance_name}.json"
    assert expected_file.exists()

def test_reproduce_sensitivity_analysis_smoke(clean_results, monkeypatch):
    """Smoke test for 'reproduce-paper sensitivity-analysis'."""
    # Use one of the synthetic files
    demand_day = "synthetic_sales_2024-06-04_demand.csv"
    
    def fake_run_config(config_path, demand_path, output_dir, skip_existing=True):
        # Mimic what the real function does: write a result file and return a dict
        
        # Determine paths
        config_name = config_path.stem
        demand_name = demand_path.stem
        output_path = output_dir / f"{config_name}_{demand_name}.json"
        
        # Write dummy result file
        import json
        with open(output_path, 'w') as f:
            json.dump({"status": "success", "config": config_name, "demand_day": demand_name}, f)
            
        return {
            "config": config_name,
            "demand_day": demand_name,
            "status": "success",
            "total_cost": 100.0,
            "total_vehicles": 5,
            "total_fixed_cost": 50.0,
            "total_variable_cost": 50.0,
            "num_customers": 10,
            "runtime_sec": 0.1,
            "parameter": config_path.parent.name # This is usually added by the caller, but useful for debug
        }

    # We need to patch it where it is defined
    monkeypatch.setattr("fleetmix.experiments.reproduce_paper.sensitivity_runner.run_config_on_demand_day", fake_run_config)
    
    # Also mock logging to avoid clutter
    monkeypatch.setattr("fleetmix.utils.logging.log_error", lambda x: print(f"LOG_ERROR: {x}"))
    
    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "sensitivity-analysis",
            "--demand-days",
            demand_day,
            "--parameters",
            "capacity", # Only one parameter
            "--variations",
            "baseline", # Only baseline variation
            "--fleet-types",
            "scv", # Only one fleet type
            "--output",
            str(clean_results),
        ],
    )
    
    assert result.exit_code == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    
    # Check structure: results/baseline/baseline_scv/baseline_scv_synthetic_sales_...json
    # Note: 'baseline' is treated as a special parameter group in the runner logic
    expected_file = clean_results / "baseline" / "baseline_scv" / f"baseline_scv_{demand_day.replace('.csv', '')}.json"
    
    if not expected_file.exists():
        print("\nFiles in clean_results:")
        for f in clean_results.rglob("*"):
            print(f"  - {f}")
            
    assert expected_file.exists()

def test_reproduce_fleet_composition_smoke(clean_results, monkeypatch):
    """Smoke test for 'reproduce-paper fleet-composition'."""
    demand_day = "synthetic_sales_2024-06-04_demand.csv"
    
    # Mock `_collect_day_summary` which calls optimize
    def fake_collect_day_summary(demand_path, params, fleet_label, alpha, C):
        return {
            "instance": demand_path.stem,
            "fleet_type": fleet_label,
            "alpha": float(alpha),
            "C": float(C),
            "total_cost": 100.0,
            "mcv_share": 0.0,
            "delta_cost_pct_vs_scv": 0.0,
            "avg_visits_per_customer": 1.0,
            "cost_per_drop": 10.0,
            "cost_per_kg": 1.0,
            "split_rate": 0.0,
            "alpha_surcharge_pct": 0.0,
            "c_pct_scv": 0.0,
            "total_fixed_cost": 50.0,
            "total_variable_cost": 50.0,
            "total_penalties": 0.0,
            "total_light_load_penalties": 0.0,
            "total_compartment_penalties": 0.0,
            "total_vehicles": 5,
            "vehicles_used": {"SCV": 5},
            "solver_runtime_sec": 0.1,
            "optimality_gap": 0.0,
            "total_route_time_hours": 10.0,
            "total_distance_km": 100.0,
            "num_customers": 10,
            "total_demand_kg": 1000.0
        }
    
    monkeypatch.setattr("fleetmix.experiments.reproduce_paper.fleet_composition_runner._collect_day_summary", fake_collect_day_summary)
    
    # We also need to mock `make_scv_fleet` and `make_mixed_fleet`
    class DummyParams:
        class runtime:
            debug = False
            
    monkeypatch.setattr("fleetmix.experiments.reproduce_paper.fleet_composition_runner.make_scv_fleet", lambda *args, **kwargs: DummyParams())
    monkeypatch.setattr("fleetmix.experiments.reproduce_paper.fleet_composition_runner.make_mixed_fleet", lambda *args, **kwargs: DummyParams())

    result = runner.invoke(
        app,
        [
            "reproduce-paper",
            "fleet-composition",
            "--demand-days",
            demand_day,
            "--alpha-grid",
            "1.0", # Single point
            "--c-values",
            "0", # Single point
            "--output",
            str(clean_results),
        ],
    )
    
    assert result.exit_code == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    
    # Check for SCV baseline (runs automatically)
    scv_file = clean_results / "raw" / f"{demand_day.replace('.csv', '')}_SCV_BASE.json"
    assert scv_file.exists()
    
    # Check for Mixed run
    mixed_file = clean_results / "raw" / f"{demand_day.replace('.csv', '')}_MIXED_1.00_0.json"
    assert mixed_file.exists()
    
    # Check for summary
    assert (clean_results / "summary_mixed.parquet").exists()
    assert (clean_results / "summary_mixed.csv").exists()
