import pandas as pd
import pytest
from unittest.mock import patch
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
        'fleetmix.pipeline.vrp_interface.solve_fsm_problem',
        lambda *args, **kw: DummySolution()
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


def test_run_optimization_prints_and_returns():
    df = pd.DataFrame()
    params = DummyParams()
    
    # Mock the logging functions to verify they're called
    with patch('fleetmix.pipeline.vrp_interface.log_progress') as mock_progress:
        with patch('fleetmix.pipeline.vrp_interface.log_detail') as mock_detail:
            sol, cfg = run_optimization(df, params, verbose=False)
    
    # Check that the logging functions were called with expected messages
    mock_progress.assert_called_once_with("Optimization Results:")
    assert mock_detail.call_count == 3  # Called 3 times for total cost, vehicles used, and expected vehicles
    
    # Check return values
    assert sol['total_cost'] == 0
    assert isinstance(cfg, pd.DataFrame) 