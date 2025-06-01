import pandas as pd

from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.core_types import VehicleSpec

def test_generate_vehicle_configurations_basic():
    # One vehicle type, two goods
    vehicle_types_raw = {'A': {'capacity': 10, 'fixed_cost': 5}}
    # Convert to VehicleSpec
    vehicle_types = {k: VehicleSpec(**v) for k, v in vehicle_types_raw.items()}
    goods = ['Dry', 'Frozen']
    df = generate_vehicle_configurations(vehicle_types, goods)
    # Must be DataFrame
    assert isinstance(df, pd.DataFrame)
    # Each config must have at least one compartment bit set
    assert all((df['Dry'] + df['Frozen']) >= 1)
    # Capacity and Fixed_Cost columns should match input
    assert all(df['Capacity'] == 10)
    assert all(df['Fixed_Cost'] == 5)
    # Config_IDs should be unique and start at 1
    assert df['Config_ID'].is_unique
    assert set(df.columns) >= {'Vehicle_Type', 'Config_ID', 'Capacity', 'Fixed_Cost', 'Dry', 'Frozen'} 