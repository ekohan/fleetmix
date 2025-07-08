import pandas as pd
import pytest

from fleetmix.core_types import FleetmixSolution
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.config import load_fleetmix_params
from fleetmix.benchmarking.models import InstanceSpec


@pytest.fixture
def test_params():
    """Load test parameters from config file."""
    return load_fleetmix_params("src/fleetmix/config/default_config.yaml")


@pytest.fixture(autouse=True)
def stub_everything(monkeypatch, test_params):
    # Stub converters in pipeline module to return InstanceSpec instead of ProblemParams
    test_instance_spec = InstanceSpec(
        expected_vehicles=test_params.problem.expected_vehicles,
        depot=test_params.problem.depot,
        goods=test_params.problem.goods,
        vehicles=test_params.problem.vehicles,
    )
    monkeypatch.setattr(
        "fleetmix.benchmarking.converters.cvrp.convert_cvrp_to_fsm",
        lambda *args, **kw: (pd.DataFrame(), test_instance_spec),
    )
    monkeypatch.setattr(
        "fleetmix.benchmarking.converters.mcvrp.convert_mcvrp_to_fsm",
        lambda *args, **kw: (pd.DataFrame(), test_instance_spec),
    )
    # Stub pipeline helper functions
    monkeypatch.setattr(
        "fleetmix.pipeline.vrp_interface.generate_vehicle_configurations",
        lambda *args, **kw: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "fleetmix.pipeline.vrp_interface.generate_feasible_clusters",
        lambda *args, **kw: pd.DataFrame(),
    )
    # Stub solver in pipeline
    monkeypatch.setattr(
        "fleetmix.pipeline.vrp_interface.optimize_fleet",
        lambda *args, **kw: FleetmixSolution(
            total_cost=0,
            vehicles_used={},
            selected_clusters=pd.DataFrame(),
            missing_customers=set(),
            total_fixed_cost=0,
            total_variable_cost=0,
            total_light_load_penalties=0,
            total_compartment_penalties=0,
            total_penalties=0,
            solver_name="stub",
            solver_status="Optimal",
            total_vehicles=0,
            solver_runtime_sec=0.0,
            time_measurements=None,
        ),
    )
    # Stub post-optimization in pipeline
    monkeypatch.setattr(
        "fleetmix.pipeline.vrp_interface.improve_solution",
        lambda solution, *args, **kw: solution,  # Just return the same solution
    )
    yield


def test_convert_to_fsm_cvrp():
    from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType

    df, instance_spec = convert_to_fsm(
        VRPType.CVRP,
        instance_names=["foo"],
        benchmark_type=CVRPBenchmarkType.NORMAL,
        num_goods=2,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(instance_spec, InstanceSpec)
    assert hasattr(instance_spec, 'expected_vehicles')
    assert hasattr(instance_spec, 'depot')
    assert hasattr(instance_spec, 'goods')
    assert hasattr(instance_spec, 'vehicles')


def test_convert_to_fsm_mcvrp():
    df, instance_spec = convert_to_fsm(VRPType.MCVRP, instance_path="dummy")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(instance_spec, InstanceSpec)
    assert hasattr(instance_spec, 'expected_vehicles')
    assert hasattr(instance_spec, 'depot')
    assert hasattr(instance_spec, 'goods')
    assert hasattr(instance_spec, 'vehicles')


def test_run_optimization_prints_and_returns(caplog, test_params):
    df = pd.DataFrame()
    sol = run_optimization(df, test_params)
    cfg = sol.configurations

    # Check that the logging message appears in the captured logs
    assert any("Optimization Results:" in record.message for record in caplog.records)
    assert sol.total_cost == 0
    assert isinstance(cfg, pd.DataFrame)


def test_run_optimization_with_post_optimization(caplog, monkeypatch, test_params):
    """Test that post-optimization is called when params.post_optimization is True"""
    post_opt_called = False
    
    def mock_improve_solution(solution, *args, **kw):
        nonlocal post_opt_called
        post_opt_called = True
        return solution
    
    monkeypatch.setattr(
        "fleetmix.pipeline.vrp_interface.improve_solution",
        mock_improve_solution,
    )
    
    df = pd.DataFrame()
    # test_params already has post_optimization=True by default
    
    sol = run_optimization(df, test_params)
    cfg = sol.configurations
    
    assert post_opt_called, "Post-optimization should have been called when enabled"
    assert sol.total_cost == 0
    assert isinstance(cfg, pd.DataFrame)



