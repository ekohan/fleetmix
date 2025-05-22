import itertools
import pandas as pd

def generate_vehicle_configurations(vehicle_types, goods):
    """Enumerate every feasible vehicle–compartment configuration.

    For each vehicle *type* and for every non-empty subset of compartments, a
    new configuration is created.  The Cartesian product is represented using
    a binary vector where each element corresponds to whether a good can be
    carried in that configuration.

    Args:
        vehicle_types (dict): Mapping ``{vehicle_name: {'capacity': int,
            'fixed_cost': float}}``.
        goods (list[str]): Ordered list of product types (e.g., ``['Dry',
            'Chilled', 'Frozen']``).  The order must stay consistent across the
            entire codebase.

    Returns:
        pd.DataFrame: One row per configuration with columns
        ``goods``, ``Vehicle_Type``, ``Config_ID``, ``Capacity``, ``Fixed_Cost``.

    Example:
        >>> vehicle_types = {'A': {'capacity': 3000, 'fixed_cost': 100}}
        >>> configs = generate_vehicle_configurations(vehicle_types, ['Dry', 'Frozen'])
        >>> len(configs)
        3  # (Dry) (Frozen) (Dry+Frozen)
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