# tests/core/test_fsm_scenarios.py

import pandas as pd
import pytest
from fleetmix.optimization import optimize_fleet_selection
from fleetmix.config.parameters import Parameters

# Define geographic coordinates for test customers
CUSTOMER_COORDS = {
    "C1": (40.7128, -74.0060),  # New York City
    "C2": (34.0522, -118.2437), # Los Angeles
    "C3": (41.8781, -87.6298),  # Chicago
    "C4": (29.7604, -95.3698),  # Houston
    "C5": (39.9526, -75.1652),  # Philadelphia
    # Add more if other scenarios use different customer IDs
}

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions: (name, clusters, configs, params_updates, expected)
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    (
        "S0_happy",
        # clusters
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 1, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        # configs
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            }
        ],
        # parameter overrides
        {"variable_cost_per_hour": 1},
        # expected results
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 6,
        },
    ),
    (
        "S1_size_filter",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 8, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 2,
                "Fixed_Cost": 2,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 6,
        },
    ),
    (
        "S2_cheapest",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 3, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 4,
                "Fixed_Cost": 4,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"B": 1},
            "total_fixed_cost": 4,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 5,
        },
    ),
    (
        "S3_light_load",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 3, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            }
        ],
        {
            "variable_cost_per_hour": 1,
            "light_load_penalty": 100,
            "light_load_threshold": 0.5,
        },
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 100,
            "total_compartment_penalties": 0,
            "total_cost": 106,
        },
    ),
    (
        "S4_compartment_penalty",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 1, "Chilled": 1, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 1,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            }
        ],
        {
            "variable_cost_per_hour": 1,
            "compartment_setup_cost": 50,
        },
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 50,
            "total_cost": 56,
        },
    ),
    (
        "S5_infeasible",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 5,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            }
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": {"C1"},
            "vehicles_used": {},
            "total_fixed_cost": 0,
            "total_variable_cost": 0,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 0,
        },
    ),
    (
        "S6_two_clusters",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 2, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            },
            {
                "Cluster_ID": 2,
                "Customers": ["C2"],
                "Total_Demand": {"Dry": 9, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            },
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 5,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 10,
                "Fixed_Cost": 8,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
                "avg_speed": 30.0,
                "service_time": 25.0,
                "max_route_time": 10.0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1, "B": 1},
            "total_fixed_cost": 13,
            "total_variable_cost": 2,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 15,
        },
    ),
]

@pytest.mark.parametrize(["name", "clusters", "configs", "upd", "exp"], SCENARIOS)
def test_fsm_scenarios(name, clusters, configs, upd, exp):
    # --- Start Change 2 ---
    # Calculate and add Centroid_Latitude/Longitude to each cluster dict
    # Also, gather all unique customer IDs for customers_df creation
    all_customer_ids_in_scenario = set()
    for cluster_dict in clusters:
        if not cluster_dict["Customers"]: # Handle empty customer lists
            cluster_dict["Centroid_Latitude"] = None # Or some default, e.g., depot
            cluster_dict["Centroid_Longitude"] = None
            continue

        lats = []
        lons = []
        for cust_id in cluster_dict["Customers"]:
            all_customer_ids_in_scenario.add(cust_id)
            if cust_id in CUSTOMER_COORDS:
                lats.append(CUSTOMER_COORDS[cust_id][0])
                lons.append(CUSTOMER_COORDS[cust_id][1])
            else:
                # Fallback for any customer ID not in CUSTOMER_COORDS (should not happen if defined properly)
                lats.append(0) # Or raise an error
                lons.append(0)
        
        cluster_dict["Centroid_Latitude"] = sum(lats) / len(lats) if lats else None
        cluster_dict["Centroid_Longitude"] = sum(lons) / len(lons) if lons else None

    # Build DataFrames
    clusters_df = pd.DataFrame(clusters)
    config_df   = pd.DataFrame(configs)
    
    # Ensure all customers mentioned in clusters are in customers_df
    # Create a comprehensive list of all customers that need coordinates
    # This ensures customers_df is built correctly even if a customer isn't in the first cluster
    customer_data_for_df = []
    
    # Get all unique customer IDs from all clusters in the current scenario
    # This was populated in the loop above. If a cluster had no customers,
    # those scenarios might not list any customers here.
    # We need to get customers from the original 'clusters' list definition for safety.
    unique_customers_in_scenario_clusters = set()
    for c_dict in clusters: # Use the original 'clusters' list as passed to the test
        for cust_id in c_dict.get("Customers", []):
            unique_customers_in_scenario_clusters.add(cust_id)

    # If 'missing_customers' is in exp, those customers also need to be in customers_df
    if "missing_customers" in exp and exp["missing_customers"]:
        for cust_id in exp["missing_customers"]:
            unique_customers_in_scenario_clusters.add(cust_id)
            
    if not unique_customers_in_scenario_clusters and name == "S12_no_clusters_no_customers": # Special case from one of the scenarios
        pass # No customers to add
    elif not unique_customers_in_scenario_clusters:
        # If still no customers, this might be an issue with a scenario definition
        # or a scenario that genuinely has no customers and no expected missing ones.
        # For scenarios like S0_happy, C1 is defined.
        # Let's check if the original clusters list was empty.
        if not clusters: # If the input 'clusters' list for the scenario was empty
            pass # No customers to add for customers_df from cluster data
        # else:
            # This case might indicate an issue if clusters were defined but had no customers,
            # and no customers were expected to be missing.
            # For now, we assume valid scenarios or scenarios where this is intended.


    for cust_id_for_df in sorted(list(unique_customers_in_scenario_clusters)):
        lat, lon = CUSTOMER_COORDS.get(cust_id_for_df, (None, None)) # Fallback for safety
        
        # Infer demand from the first cluster that contains this customer
        # This part is tricky as customers_df needs demands, but cluster definition might be complex.
        # The original code iterated through clusters to build customers_df.
        # Let's find the original way demands were assigned for `customers_df` and adapt.
        # Original:
        # customers_df = pd.DataFrame([
        #     {
        #         "Customer_ID": c,
        #         "Dry_Demand": row["Total_Demand"].get("Dry", 0),
        #         "Chilled_Demand": row["Total_Demand"].get("Chilled", 0),
        #         "Frozen_Demand": row["Total_Demand"].get("Frozen", 0),
        #     }
        #     for row in clusters for c in row["Customers"] # 'clusters' here is the list of dicts
        # ])
        # We need to ensure demands are correctly sourced for each customer.
        # The simplest way to keep original demand logic is to iterate through original cluster defs.
        
        # This part becomes redundant if we reconstruct customers_df as original, then add lat/lon.
        # For now, let's keep the original structure for customer_df creation for demands, then merge/add lat/lon.
        pass


    # Customers=index of clusters - Rebuild customers_df as original for demands, then add lat/lon
    customers_list_for_df = []
    # The 'clusters' variable here is the one modified with Centroids.
    # We should use the 'clusters' variable as passed into the function for demand consistency.
    # Let's rename the input parameter to avoid confusion
    # No, 'clusters' is the correct one, it's the list of dicts for the current scenario.
    
    temp_customer_demands = {} # Store demands keyed by Customer_ID
    for cl_dict in clusters: # Iterate over the scenario's cluster list
        for cust_id in cl_dict.get("Customers", []):
            if cust_id not in temp_customer_demands: # Take demand from first encounter
                 temp_customer_demands[cust_id] = {
                    "Dry_Demand": cl_dict["Total_Demand"].get("Dry", 0),
                    "Chilled_Demand": cl_dict["Total_Demand"].get("Chilled", 0),
                    "Frozen_Demand": cl_dict["Total_Demand"].get("Frozen", 0),
                }

    # Also add customers from exp["missing_customers"] if any, with zero demand.
    if "missing_customers" in exp:
        for cust_id in exp["missing_customers"]:
            if cust_id not in temp_customer_demands:
                 temp_customer_demands[cust_id] = {
                    "Dry_Demand": 0, "Chilled_Demand": 0, "Frozen_Demand": 0
                }
                
    customer_data_for_df_rebuilt = []
    for cust_id_key, demands in temp_customer_demands.items():
        lat, lon = CUSTOMER_COORDS.get(cust_id_key, (None, None)) # Default to None if not in map
        customer_entry = {
            "Customer_ID": cust_id_key,
            "Latitude": lat,
            "Longitude": lon,
            **demands # Add Dry_Demand, Chilled_Demand, Frozen_Demand
        }
        customer_data_for_df_rebuilt.append(customer_entry)
        
    customers_df = pd.DataFrame(customer_data_for_df_rebuilt)
    # Ensure 'Customer_ID' is present even if empty, for schema consistency
    if 'Customer_ID' not in customers_df.columns and not customer_data_for_df_rebuilt:
        customers_df = pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude', 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand'])
    elif not customer_data_for_df_rebuilt: # Handle case where customers_df could be empty but still needs other cols
        customers_df = pd.DataFrame(columns=['Customer_ID', 'Latitude', 'Longitude', 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand'])


    # Load & override params
    params = Parameters.from_yaml()
    for k,v in upd.items():
        setattr(params, k, v)
    # Solve or validate infeasible
    from fleetmix.optimization import _create_model, _extract_solution, _validate_solution
    if exp["missing_customers"]:
        # Infeasible: model should inject NoVehicle and warn
        model, y_vars, x_vars, c_vk = _create_model(
            clusters_df, config_df, params
        )
        selected = _extract_solution(clusters_df, y_vars, x_vars)
        missing = _validate_solution(
            selected, customers_df, config_df
        )
        assert missing == exp["missing_customers"]
    else:
        result = optimize_fleet_selection(
            clusters_df, config_df, customers_df, params, verbose=False
        )
        # Compare expected
        assert result.missing_customers == exp["missing_customers"]
        assert result.vehicles_used       == exp["vehicles_used"]
        assert result.total_fixed_cost    == exp["total_fixed_cost"]
        assert pytest.approx(result.total_variable_cost) == exp["total_variable_cost"]
        assert result.total_light_load_penalties  == exp["total_light_load_penalties"]
        assert result.total_compartment_penalties == exp["total_compartment_penalties"]
        assert result.total_cost          == exp["total_cost"]