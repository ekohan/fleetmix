import pandas as pd
import pytest

from fleetmix.benchmarking.converters.vrp import VRPType, convert_vrp_to_fsm as convert_to_fsm
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


