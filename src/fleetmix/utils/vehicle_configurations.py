import itertools
import pandas as pd

def generate_vehicle_configurations(vehicle_types, goods):
    """
    Enumerate every feasible vehicle–compartment combination (paper §4.4).
    """
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    compartment_configs = []
    config_id = 1
    
    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
                continue
            compartment = dict(zip(goods, option))
            compartment['Vehicle_Type'] = vt_name
            compartment['Config_ID'] = config_id
            compartment['Capacity'] = vt_info['capacity']
            compartment['Fixed_Cost'] = vt_info['fixed_cost']
            compartment_configs.append(compartment)
            config_id += 1

    return pd.DataFrame(compartment_configs)