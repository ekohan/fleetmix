import hypothesis.strategies as st
from hypothesis import given, settings

from fleetmix.core_types import VehicleConfiguration, VehicleSpec
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations


@settings(max_examples=20)
@given(
    vehicle_types_raw=st.dictionaries(
        keys=st.text(min_size=1, max_size=3),
        values=st.fixed_dictionaries(
            {
                "capacity": st.integers(min_value=1, max_value=100),
                "fixed_cost": st.integers(min_value=0, max_value=1000),
            }
        ),
        min_size=1,
        max_size=3,
    ),
    goods=st.lists(
        st.text(min_size=1, max_size=3), min_size=1, max_size=3, unique=True
    ),
)
def test_generate_vehicle_configurations_hypothesis(vehicle_types_raw, goods):
    """Test vehicle configuration generation with varied inputs using Hypothesis."""
    # Convert raw dicts to VehicleSpec objects
    vehicle_types = {
        name: VehicleSpec(**specs) for name, specs in vehicle_types_raw.items()
    }

    configs = generate_vehicle_configurations(vehicle_types, goods)

    # Must be a List[VehicleConfiguration]
    assert isinstance(configs, list)
    assert all(isinstance(config, VehicleConfiguration) for config in configs)

    # Config_ID unique and sequential starting at "1"
    config_ids = [config.config_id for config in configs]
    assert len(config_ids) == len(set(config_ids))  # All unique
    # Convert to int for sorting, then back to string for comparison
    expected_ids = [str(i) for i in range(1, len(config_ids) + 1)]
    assert sorted(config_ids, key=int) == expected_ids

    # Every configuration has at least one compartment enabled
    assert all(any(config.compartments.values()) for config in configs)

    # Capacity and Fixed_Cost match input values per configuration
    for config in configs:
        vt = config.vehicle_type
        assert config.capacity == vehicle_types[vt]["capacity"]
        assert config.fixed_cost == vehicle_types[vt]["fixed_cost"]
