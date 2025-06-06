import pandas as pd
import pytest

from fleetmix.clustering import _is_customer_feasible, get_cached_demand
from fleetmix.core_types import VehicleConfiguration


def test_is_customer_feasible_all_goods_fit():
    # Customer with dry demand 5, chilled 0, frozen 0 and config that supports dry
    customer = pd.Series({'Dry_Demand':5, 'Chilled_Demand':0, 'Frozen_Demand':0, 'Customer_ID':'C1'})
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type='TestVehicle',
        capacity=10,
        fixed_cost=100,
        compartments={'Dry': True, 'Chilled': True, 'Frozen': True}
    )
    goods = ['Dry', 'Chilled', 'Frozen']
    assert _is_customer_feasible(customer, config, goods)


def test_is_customer_feasible_missing_compartment():
    # Customer with chilled demand but config has no chilled
    customer = pd.Series({'Dry_Demand':0, 'Chilled_Demand':3, 'Frozen_Demand':0, 'Customer_ID':'C2'})
    config = VehicleConfiguration(
        config_id=2,
        vehicle_type='TestVehicle',
        capacity=10,
        fixed_cost=100,
        compartments={'Dry': True, 'Chilled': False, 'Frozen': True}
    )
    goods = ['Dry', 'Chilled', 'Frozen']
    assert not _is_customer_feasible(customer, config, goods)


def test_is_customer_feasible_capacity_exceeded():
    # Customer dry demand 15, config capacity 10 => not feasible
    customer = pd.Series({'Dry_Demand':15, 'Chilled_Demand':0, 'Frozen_Demand':0, 'Customer_ID':'C3'})
    config = VehicleConfiguration(
        config_id=3,
        vehicle_type='TestVehicle',
        capacity=10,
        fixed_cost=100,
        compartments={'Dry': True, 'Chilled': True, 'Frozen': True}
    )
    goods = ['Dry', 'Chilled', 'Frozen']
    assert not _is_customer_feasible(customer, config, goods)


def test_get_cached_demand_consistent_and_correct():
    # Create customer DataFrame with two rows
    goods = ['Dry', 'Chilled']
    df = pd.DataFrame([
        {'Latitude':0,'Longitude':0,'Dry_Demand':2,'Chilled_Demand':1,'Customer_ID':'C1'},
        {'Latitude':1,'Longitude':1,'Dry_Demand':3,'Chilled_Demand':2,'Customer_ID':'C2'}
    ])
    cache = {}
    # First call
    d1 = get_cached_demand(df, goods, cache)
    # Should sum demands correctly
    assert d1['Dry'] == 5
    assert d1['Chilled'] == 3
    # Second call should retrieve from cache and equal
    d2 = get_cached_demand(df, goods, cache)
    assert d1 is d2 or d1 == d2
    # Cache should have an entry
    assert len(cache) == 1 