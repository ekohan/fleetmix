import hypothesis.strategies as st
from hypothesis import given, settings
import pandas as pd
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations

@settings(max_examples=20)
@given(
    vehicle_types=st.dictionaries(
        keys=st.text(min_size=1, max_size=3),
        values=st.fixed_dictionaries({
            'capacity': st.integers(min_value=1, max_value=100),
            'fixed_cost': st.integers(min_value=0, max_value=1000)
        }),
        min_size=1, max_size=3
    ),
    goods=st.lists(st.text(min_size=1, max_size=3), min_size=1, max_size=3, unique=True)
)
def test_generate_vehicle_configurations_hypothesis(vehicle_types, goods):
    df = generate_vehicle_configurations(vehicle_types, goods)
    # Must be a DataFrame
    assert isinstance(df, pd.DataFrame)
    # Config_ID unique and sequential starting at 1
    ids = df['Config_ID'].tolist()
    assert len(ids) == len(set(ids))
    assert sorted(ids) == list(range(1, len(ids)+1))
    # Every row has at least one compartment bit set
    sum_bits = df[goods].sum(axis=1)
    assert all(sum_bits >= 1)
    # Capacity and Fixed_Cost columns match input values per row
    for _, row in df.iterrows():
        vt = row['Vehicle_Type']
        assert row['Capacity'] == vehicle_types[vt]['capacity']
        assert row['Fixed_Cost'] == vehicle_types[vt]['fixed_cost'] 