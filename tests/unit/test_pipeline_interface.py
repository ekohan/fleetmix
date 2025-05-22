import pandas as pd
import pytest
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization

class DummyParams:
    def __init__(self):
        self.vehicles = {}
        self.goods = {}
        self.expected_vehicles = 3

class DummySolution(dict):
    def __init__(self):
        super().__init__({
            'total_cost': 0,
            'vehicles_used': {},
            'selected_clusters': pd.DataFrame(),
            'missing_customers': set(),
            'total_fixed_cost': 0,
            'total_variable_cost': 0,
            'total_light_load_penalties': 0,
            'total_compartment_penalties': 0,
            'total_penalties': 0,
            'solver_name': 'stub',
            'solver_status': 'Optimal'
        })

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
        'fleetmix.pipeline.vrp_interface._optimization_module.solve_fsm_problem',
        lambda *args, **kw: DummySolution()
    )
    yield


def test_convert_to_fsm_cvrp():
    df, params = convert_to_fsm(
        VRPType.CVRP,
        instance_names=['foo'],
        benchmark_type=None,
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


def test_run_optimization_prints_and_returns(capsys):
    df = pd.DataFrame()
    params = DummyParams()
    sol, cfg = run_optimization(df, params, verbose=False)
    out = capsys.readouterr().out
    assert 'Optimization Results:' in out
    assert sol['total_cost'] == 0
    assert isinstance(cfg, pd.DataFrame) 