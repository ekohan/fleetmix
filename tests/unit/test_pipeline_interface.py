import pandas as pd
import pytest
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.core_types import FleetmixSolution

class DummyParams:
    def __init__(self):
        self.vehicles = {}
        self.goods = {}
        self.expected_vehicles = 3
        self.allow_split_stops = False

@pytest.fixture(autouse=True)
def stub_everything(monkeypatch):
    # Stub converters in pipeline module
    monkeypatch.setattr(
        'fleetmix.benchmarking.converters.cvrp.convert_cvrp_to_fsm',
        lambda *args, **kw: (pd.DataFrame(), DummyParams())
    )
    monkeypatch.setattr(
        'fleetmix.benchmarking.converters.mcvrp.convert_mcvrp_to_fsm',
        lambda *args, **kw: (pd.DataFrame(), DummyParams())
    )
    # Stub pipeline helper functions
    monkeypatch.setattr(
        'fleetmix.pipeline.vrp_interface.generate_vehicle_configurations',
        lambda *args, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        'fleetmix.pipeline.vrp_interface.generate_clusters_for_configurations',
        lambda *args, **kw: pd.DataFrame()
    )
    # Stub solver in pipeline
    monkeypatch.setattr(
        'fleetmix.pipeline.vrp_interface.solve_fsm_problem',
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
            solver_name='stub',
            solver_status='Optimal',
            total_vehicles=0,
            solver_runtime_sec=0.0,
            time_measurements=None
        )
    )
    yield


def test_convert_to_fsm_cvrp():
    from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType
    df, params = convert_to_fsm(
        VRPType.CVRP,
        instance_names=['foo'],
        benchmark_type=CVRPBenchmarkType.NORMAL,
        num_goods=2
    )
    assert isinstance(df, pd.DataFrame)
    assert hasattr(params, 'expected_vehicles')


def test_convert_to_fsm_mcvrp():
    df, params = convert_to_fsm(
        VRPType.MCVRP,
        instance_path='dummy'
    )
    assert isinstance(df, pd.DataFrame)
    assert hasattr(params, 'expected_vehicles')


def test_run_optimization_prints_and_returns(caplog):
    df = pd.DataFrame()
    params = DummyParams()
    sol, cfg = run_optimization(df, params, verbose=False)
    
    # Check that the logging message appears in the captured logs
    assert any('Optimization Results:' in record.message for record in caplog.records)
    assert sol.total_cost == 0
    assert isinstance(cfg, pd.DataFrame) 