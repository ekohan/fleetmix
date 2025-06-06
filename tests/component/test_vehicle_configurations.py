import pandas as pd

from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.core_types import VehicleSpec, VehicleConfiguration

def test_generate_vehicle_configurations_basic():
    # One vehicle type, two goods
    vehicle_types_raw = {'A': {'capacity': 10, 'fixed_cost': 5}}
    # Convert to VehicleSpec
    vehicle_types = {k: VehicleSpec(**v) for k, v in vehicle_types_raw.items()}
    goods = ['Dry', 'Frozen']
    configs = generate_vehicle_configurations(vehicle_types, goods)
    
    # Must be List[VehicleConfiguration]
    assert isinstance(configs, list)
    assert all(isinstance(config, VehicleConfiguration) for config in configs)
    
    # Each config must have at least one compartment enabled
    assert all(any(config.compartments.values()) for config in configs)
    
    # Capacity and Fixed_Cost should match input
    assert all(config.capacity == 10 for config in configs)
    assert all(config.fixed_cost == 5 for config in configs)
    
    # Config_IDs should be unique and start at 1
    config_ids = [config.config_id for config in configs]
    assert len(set(config_ids)) == len(config_ids)  # All unique
    assert min(config_ids) == 1
    
    # Should have expected compartment combinations (excluding all-zeros)
    # With 2 goods, we have 2^2 - 1 = 3 valid combinations: (1,0), (0,1), (1,1)
    assert len(configs) == 3
    
    # Check that all expected fields are accessible via bracket notation
    config = configs[0]
    assert config['Vehicle_Type'] == 'A'
    assert config['Config_ID'] == config.config_id
    assert config['Capacity'] == 10
    assert config['Fixed_Cost'] == 5
    assert config['Dry'] in [0, 1]
    assert config['Frozen'] in [0, 1] 