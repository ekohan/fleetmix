import pandas as pd
from fleetmix.utils.vehicle_configurations import _generate_vehicle_configurations_df
from fleetmix.internal_types import VehicleSpec

def test_generate_vehicle_configurations_basic():
    """Test basic vehicle configuration generation."""
    vehicle_types = {
        'A': VehicleSpec(capacity=10, fixed_cost=5.0, avg_speed=30.0, service_time=25.0, max_route_time=10.0),
        'B': VehicleSpec(capacity=20, fixed_cost=10.0, avg_speed=30.0, service_time=25.0, max_route_time=10.0)
    }
    goods = ['Dry', 'Frozen']
    
    df = _generate_vehicle_configurations_df(vehicle_types, goods)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check columns
    expected_cols = ['Config_ID', 'Vehicle_Type', 'Capacity', 'Fixed_Cost', 'Dry', 'Frozen']
    for col in expected_cols:
        assert col in df.columns
    
    # Check that we have configurations for both vehicle types
    assert set(df['Vehicle_Type'].unique()) == {'A', 'B'}
    
    # Check that we have all non-empty compartment combinations
    # For 2 goods, we should have 3 combinations per vehicle type: (0,1), (1,0), (1,1)
    assert len(df) == 6  # 2 vehicle types * 3 combinations 