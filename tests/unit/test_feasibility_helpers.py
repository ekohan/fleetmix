import pytest
import pandas as pd

from fleetmix.clustering.generator import _is_customer_feasible
from fleetmix.clustering.heuristics import get_cached_demand
from fleetmix.core_types import CapacitatedClusteringContext, Customer, VehicleConfiguration


def test_is_customer_feasible_all_goods_fit():
    # Customer with dry demand 5, chilled 0, frozen 0 and config that supports dry
    customer = Customer(
        customer_id="C1",
        demands={"dry": 5, "chilled": 0, "frozen": 0},
        location=(0.0, 0.0),
        service_time=25.0,
    )
    config = VehicleConfiguration(
        config_id=1,
        vehicle_type="TestVehicle",
        capacity=10,
        fixed_cost=100,
        compartments={"dry": True, "chilled": True, "frozen": True},
    )
    goods = ["dry", "chilled", "frozen"]
    assert _is_customer_feasible(customer, config, goods)


def test_is_customer_feasible_missing_compartment():
    # Customer with chilled demand but config has no chilled
    customer = Customer(
        customer_id="C2",
        demands={"dry": 0, "chilled": 3, "frozen": 0},
        location=(0.0, 0.0),
        service_time=25.0,
    )
    config = VehicleConfiguration(
        config_id=2,
        vehicle_type="TestVehicle",
        capacity=10,
        fixed_cost=100,
        compartments={"dry": True, "chilled": False, "frozen": True},
    )
    goods = ["dry", "chilled", "frozen"]
    assert not _is_customer_feasible(customer, config, goods)


def test_is_customer_feasible_capacity_exceeded():
    # Customer dry demand 15, config capacity 10 => not feasible
    customer = Customer(
        customer_id="C3",
        demands={"dry": 15, "chilled": 0, "frozen": 0},
        location=(0.0, 0.0),
        service_time=25.0,
    )
    config = VehicleConfiguration(
        config_id=3,
        vehicle_type="TestVehicle",
        capacity=10,
        fixed_cost=100,
        compartments={"dry": True, "chilled": True, "frozen": True},
    )
    goods = ["dry", "chilled", "frozen"]
    assert not _is_customer_feasible(customer, config, goods)


def test_get_cached_demand_consistent_and_correct():
    # Create Customer objects
    goods = ["dry", "chilled"]
    customers = [
        Customer(
            customer_id="C1",
            demands={"dry": 2, "chilled": 1},
            location=(0.0, 0.0),
            service_time=25.0,
        ),
        Customer(
            customer_id="C2",
            demands={"dry": 3, "chilled": 2},
            location=(1.0, 1.0),
            service_time=25.0,
        ),
    ]
    
    cache = {}
    # First call
    d1 = get_cached_demand(customers, goods, cache)
    # Should sum demands correctly
    assert d1["dry"] == 5
    assert d1["chilled"] == 3
    # Second call should retrieve from cache and equal
    d2 = get_cached_demand(customers, goods, cache)
    assert d1 is d2 or d1 == d2
    # Cache should have an entry
    assert len(cache) == 1
