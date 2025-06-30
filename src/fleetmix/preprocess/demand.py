"""
Demand preprocessing for split-stop capability.

This module provides functions to handle split-stop scenarios by creating
pseudo-customers that represent subsets of goods that a physical customer requires.

TODO: rename?
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from fleetmix.core_types import Customer, CustomerBase, PseudoCustomer
from fleetmix.utils.common import build_zero_filled_demands
from fleetmix.utils.logging import FleetmixLogger

if TYPE_CHECKING:
    from fleetmix.core_types import VehicleConfiguration
# TODO fix nasty import.

logger = FleetmixLogger.get_logger(__name__)


def get_feasible_goods_combinations(
    goods_with_demand: list[str], configurations: list["VehicleConfiguration"]
) -> list[tuple[str, ...]]:
    """
    Get all feasible combinations of goods that can be served by at least one vehicle.

    Args:
        goods_with_demand: List of goods that have positive demand
        configurations: List of vehicle configurations

    Returns:
        List of tuples, each representing a feasible combination of goods
    """
    feasible_combinations = set()

    # For each configuration, add all possible subsets of its compartments
    # that overlap with the goods_with_demand
    for config in configurations:
        # Get goods this vehicle can carry that are also demanded
        vehicle_goods = [
            good for good in goods_with_demand if config.compartments.get(good, False)
        ]

        if not vehicle_goods:
            continue

        # Generate all non-empty subsets of vehicle_goods
        num_goods = len(vehicle_goods)
        for mask in range(1, (1 << num_goods)):
            subset = tuple(
                vehicle_goods[i] for i in range(num_goods) if mask & (1 << i)
            )
            feasible_combinations.add(subset)

    return sorted(list(feasible_combinations))


def explode_customer_smart(
    customer_id: str,
    demands: Mapping[str, float],
    location: tuple[float, float],
    configurations: list["VehicleConfiguration"],
    service_time: float = 25.0,
) -> list[PseudoCustomer]:
    """
    Explode a customer into pseudo-customers, but only create feasible combinations
    based on vehicle configurations.

    Args:
        customer_id: Original customer ID
        demands: Dict mapping good names to demand quantities
        location: (latitude, longitude) tuple
        configurations: List of vehicle configurations to check feasibility
        service_time: Service time in minutes per stop

    Returns:
        List of PseudoCustomer objects representing feasible subsets of goods
    """
    # Get goods with positive demand
    goods_with_demand = [good for good, qty in demands.items() if qty > 0]
    assert goods_with_demand, (
        f"Customer {customer_id} has no positive demands - this should not happen"
    )

    # Get feasible combinations based on vehicle configurations
    feasible_combinations = get_feasible_goods_combinations(
        goods_with_demand, configurations
    )

    if not feasible_combinations:
        logger.warning(
            f"Customer {customer_id} cannot be served by any vehicle configuration! "
            f"Demands: {goods_with_demand}"
        )
        # Fall back to creating individual pseudo-customers for each good
        feasible_combinations = [(good,) for good in goods_with_demand]

    pseudo_customers = []
    for subset_goods in feasible_combinations:
        # Create pseudo-customer ID
        pseudo_id = f"{customer_id}::{'-'.join(subset_goods)}"

        # Build zero-filled demand vector, then populate subset goods
        demand_vector = build_zero_filled_demands(list(demands.keys()))
        for good in subset_goods:
            demand_vector[good] = float(demands[good])

        pseudo_customer = PseudoCustomer(
            customer_id=pseudo_id,
            origin_id=customer_id,
            subset=subset_goods,
            demands=demand_vector,
            location=location,
            service_time=service_time,
        )
        pseudo_customers.append(pseudo_customer)

    logger.debug(
        f"Exploded customer {customer_id} into {len(pseudo_customers)} feasible pseudo-customers"
    )
    return pseudo_customers


def explode_customer(
    customer_id: str,
    demands: Mapping[str, float],
    location: tuple[float, float],
    service_time: float = 25.0,
) -> list[PseudoCustomer]:
    """
    Explode a single customer into pseudo-customers representing all possible good subsets.

    Args:
        customer_id: Original customer ID
        demands: Dict mapping good names to demand quantities
        location: (latitude, longitude) tuple
        service_time: Service time in minutes per stop

    Returns:
        List of PseudoCustomer objects representing all non-empty subsets of goods

    Example:
        >>> explode_customer("C001", {"dry": 10, "chilled": 5}, (40.7, -74.0))
        [
            PseudoCustomer(customer_id="C001::dry", origin_id="C001",
                          subset=("dry",), demands={"dry": 10, "chilled": 0}, ...),
            PseudoCustomer(customer_id="C001::chilled", origin_id="C001",
                          subset=("chilled",), demands={"dry": 0, "chilled": 5}, ...),
            PseudoCustomer(customer_id="C001::dry-chilled", origin_id="C001",
                          subset=("dry", "chilled"), demands={"dry": 10, "chilled": 5}, ...)
        ]
    """
    # Get goods with positive demand
    goods_with_demand = [good for good, qty in demands.items() if qty > 0]
    assert goods_with_demand, (
        f"Customer {customer_id} has no positive demands - this should not happen"
    )

    # Generate all non-empty subsets using bit masks
    num_goods = len(goods_with_demand)
    pseudo_customers = []

    for mask in range(1, (1 << num_goods)):  # 1 to 2^n - 1 (all non-empty subsets)
        subset_goods = tuple(
            goods_with_demand[i] for i in range(num_goods) if mask & (1 << i)
        )

        # Create pseudo-customer ID
        pseudo_id = f"{customer_id}::{'-'.join(subset_goods)}"

        # Create demand vector: subset goods get their demand, others get 0
        demand_vector = {}
        for good in demands.keys():
            if good in subset_goods:
                demand_vector[good] = demands[good]
            else:
                demand_vector[good] = 0.0

        pseudo_customer = PseudoCustomer(
            customer_id=pseudo_id,
            origin_id=customer_id,
            subset=subset_goods,
            demands=demand_vector,
            location=location,
            service_time=service_time,
        )
        pseudo_customers.append(pseudo_customer)

    logger.debug(
        f"Exploded customer {customer_id} into {len(pseudo_customers)} pseudo-customers"
    )
    return pseudo_customers


def explode_customers(
    customers: list[CustomerBase],
    configurations: list["VehicleConfiguration"] | None = None,
) -> list[CustomerBase]:
    """
    Explode regular customers into pseudo-customers while preserving existing pseudo-customers.

    Args:
        customers: List of CustomerBase objects (mix of Customer and PseudoCustomer)
        configurations: Optional list of vehicle configurations for smart explosion

    Returns:
        List of CustomerBase objects with regular customers exploded into pseudo-customers
    """
    result = []

    for customer in customers:
        if customer.is_pseudo_customer():
            # Already a pseudo-customer, keep as-is
            result.append(customer)
        else:
            # Regular customer, explode into pseudo-customers
            if configurations:
                # Use smart explosion that considers vehicle constraints
                pseudo_customers = explode_customer_smart(
                    customer_id=customer.customer_id,
                    demands=customer.demands,
                    location=customer.location,
                    configurations=configurations,
                    service_time=customer.service_time,
                )
            else:
                # Fall back to original explosion (all subsets)
                pseudo_customers = explode_customer(
                    customer_id=customer.customer_id,
                    demands=customer.demands,
                    location=customer.location,
                    service_time=customer.service_time,
                )
            result.extend(pseudo_customers)

    logger.debug(
        f"Exploded {len(customers)} customers into {len(result)} pseudo-customers"
    )
    return result


def maybe_explode(
    customers_df: pd.DataFrame,
    allow_split_stops: bool,
    configurations: list["VehicleConfiguration"] | None = None,
) -> pd.DataFrame:
    """
    Conditionally explode customers into pseudo-customers based on split-stop setting.

    Args:
        customers_df: DataFrame with customer data (Customer_ID, demands, location)
        allow_split_stops: If True, explode customers into pseudo-customers
        configurations: Optional list of vehicle configurations for smart explosion

    Returns:
        DataFrame with either original customers or pseudo-customers
    """
    if not allow_split_stops:
        logger.info("Split-stops disabled, returning original customer data")
        return customers_df.copy()

    logger.info(
        f"Split-stops enabled, exploding {len(customers_df)} customers into pseudo-customers"
    )

    if configurations:
        logger.info(
            f"Using smart explosion with {len(configurations)} vehicle configurations"
        )

    # Convert DataFrame to CustomerBase objects
    customers = Customer.from_dataframe(customers_df)

    # Explode customers into pseudo-customers
    exploded_customers = explode_customers(customers, configurations)

    # Convert back to DataFrame
    result_df = Customer.to_dataframe(exploded_customers)

    logger.info(
        f"Created {len(result_df)} pseudo-customers from {len(customers_df)} original customers"
    )

    # Debug: Show which customers needed multiple vehicles
    if configurations:
        multi_vehicle_customers = {}
        for customer in customers:
            pseudo_count = sum(
                1
                for pc in exploded_customers
                if pc.get_origin_id() == customer.customer_id
            )
            if pseudo_count > 1:
                multi_vehicle_customers[customer.customer_id] = pseudo_count

        if multi_vehicle_customers:
            logger.info(
                f"Customers requiring multiple vehicles: {len(multi_vehicle_customers)}"
            )
            for cid, count in list(multi_vehicle_customers.items())[:5]:
                logger.debug(f"  {cid}: {count} pseudo-customers")

    return result_df


# Legacy utility functions - DEPRECATED - Use CustomerBase methods instead
# TODO: remove these.
def is_pseudo_customer(customer_id: str) -> bool:
    """Check if a customer ID represents a pseudo-customer (contains '::').

    DEPRECATED: Use customer.is_pseudo_customer() method instead.
    """
    return "::" in customer_id


def get_origin_id(customer_id: str) -> str:
    """Extract the original customer ID from a pseudo-customer ID.

    DEPRECATED: Use customer.get_origin_id() method instead.
    """
    if is_pseudo_customer(customer_id):
        return customer_id.split("::")[0]
    return customer_id


def get_subset_from_id(customer_id: str) -> tuple[str, ...]:
    """Extract the goods subset from a pseudo-customer ID.

    DEPRECATED: Use customer.get_goods_subset() method instead.
    """
    if is_pseudo_customer(customer_id):
        subset_str = customer_id.split("::")[1]
        return tuple(subset_str.split("-"))
    # For regular customers, assume they serve all their positive-demand goods
    return tuple()


# TODO: this file can be prettier.
