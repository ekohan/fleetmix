import pandas as pd
from hypothesis import given, strategies as st
from fleetmix.utils.vehicle_configurations import _generate_vehicle_configurations_df
from fleetmix.internal_types import VehicleSpec

@given(
    vehicle_types_raw=st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.fixed_dictionaries({
            'capacity': st.integers(min_value=1, max_value=1000),
            'fixed_cost': st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
        }),
        min_size=1,
        max_size=5
    ),
    goods=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=5, unique=True)
)
def test_generate_vehicle_configurations_hypothesis(vehicle_types_raw, goods):
    """Property-based test for vehicle configuration generation."""
    # Convert raw dict to VehicleSpec objects
    vehicle_types = {}
    for k, v in vehicle_types_raw.items():
        v_copy = v.copy()  # Don't modify the original
        v_copy.setdefault('avg_speed', 30.0)
        v_copy.setdefault('service_time', 25.0)
        v_copy.setdefault('max_route_time', 10.0)
        vehicle_types[k] = VehicleSpec(**v_copy)

    df = _generate_vehicle_configurations_df(vehicle_types, goods)
    
    # Basic properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # Should have at least one configuration
    
    # Each configuration should have at least one compartment
    compartment_cols = [col for col in df.columns if col in goods]
    assert all((df[compartment_cols].sum(axis=1) >= 1).values)
    
    # Config IDs should be unique
    assert df['Config_ID'].is_unique
    
    # All vehicle types should be represented
    assert set(df['Vehicle_Type'].unique()) <= set(vehicle_types.keys()) 