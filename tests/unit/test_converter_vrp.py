from pathlib import Path
import types
import pytest

import fleetmix.benchmarking.converters.vrp as vrp_module

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "fleetmix" / "benchmarking" / "datasets"


def test_cvrp_dispatch(monkeypatch):
    captured = {}

    def fake_convert_cvrp_to_fsm(**kwargs):
        captured.update(kwargs)
        return ("df", "params")

    monkeypatch.setattr(vrp_module, "_cvrp", types.SimpleNamespace(convert_cvrp_to_fsm=fake_convert_cvrp_to_fsm))

    dummy_path = DATA_DIR / "cvrp" / "dummy.vrp"

    out = vrp_module.convert_vrp_to_fsm(
        vrp_type="cvrp",
        instance_names=["dummy"],
        instance_path=str(dummy_path),
        benchmark_type="normal",
    )
    assert out == ("df", "params")
    # Ensure custom_instance_paths injected (dict present and path correct)
    assert captured.get("custom_instance_paths") and captured["custom_instance_paths"]["dummy"].resolve() == dummy_path.resolve()


def test_mcvrp_dispatch(monkeypatch, tmp_path):
    captured = {}

    def fake_convert_mcvrp_to_fsm(*, instance_name, custom_instance_path):
        captured["path"] = custom_instance_path
        captured["name"] = instance_name
        return ("df", "params")

    monkeypatch.setattr(
        vrp_module,
        "_mcvrp",
        types.SimpleNamespace(convert_mcvrp_to_fsm=fake_convert_mcvrp_to_fsm),
    )

    dummy = tmp_path / "inst.dat"
    dummy.write_text("DATA")

    out = vrp_module.convert_vrp_to_fsm(
        vrp_type="mcvrp",
        instance_path=str(dummy),
    )
    assert out == ("df", "params")
    assert captured["path"].resolve() == (tmp_path / "inst.dat").resolve()


def test_mcvrp_missing_path():
    with pytest.raises(ValueError):
        vrp_module.convert_vrp_to_fsm(vrp_type="mcvrp")


def test_invalid_vrp_type():
    with pytest.raises(ValueError):
        vrp_module.convert_vrp_to_fsm(vrp_type="foo", instance_path="/tmp/x") 