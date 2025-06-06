"""
Demand preprocessing for split-stop capability.

This module provides functions to handle split-stop scenarios by creating
pseudo-customers that represent subsets of goods that a physical customer requires.

TODO: rename?
"""

from typing import List, Dict, Mapping, Tuple
import pandas as pd
import itertools

from fleetmix.core_types import PseudoCustomer
from fleetmix.utils.logging import FleetmixLogger

logger = FleetmixLogger.get_logger(__name__)


def explode_customer(customer_id: str, demands: Mapping[str, float], 
                    location: Tuple[float, float], service_time: float = 25.0) -> List[PseudoCustomer]:
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
    assert goods_with_demand, f"Customer {customer_id} has no positive demands - this should not happen"
    
    # Generate all non-empty subsets using bit masks
    num_goods = len(goods_with_demand)
    pseudo_customers = []
    
    for mask in range(1, (1 << num_goods)):  # 1 to 2^n - 1 (all non-empty subsets)
        subset_goods = tuple(
            goods_with_demand[i] for i in range(num_goods) 
            if mask & (1 << i)
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
            service_time=service_time
        )
        pseudo_customers.append(pseudo_customer)
    
    logger.debug(f"Exploded customer {customer_id} into {len(pseudo_customers)} pseudo-customers")
    return pseudo_customers


def maybe_explode(customers_df: pd.DataFrame, allow_split_stops: bool) -> pd.DataFrame:
    """
    Conditionally explode customers into pseudo-customers based on split-stop setting.
    
    Args:
        customers_df: DataFrame with customer data (Customer_ID, demands, location)
        allow_split_stops: If True, explode customers into pseudo-customers
    
    Returns:
        DataFrame with either original customers or pseudo-customers
    """
    if not allow_split_stops:
        logger.info("Split-stops disabled, returning original customer data")
        return customers_df.copy()
    
    logger.info(f"Split-stops enabled, exploding {len(customers_df)} customers into pseudo-customers")
    
    # Get demand columns
    demand_cols = [col for col in customers_df.columns if col.endswith('_Demand')]
    goods = [col.replace('_Demand', '').lower() for col in demand_cols]
    
    pseudo_customers_data = []
    
    for _, row in customers_df.iterrows():
        # Extract demand mapping
        demands = {}
        for good in goods:
            col_name = f'{good.title()}_Demand'
            demands[good] = row[col_name] if col_name in row else 0.0
        
        # Extract location and service time
        location = (row.get('Latitude', 0.0), row.get('Longitude', 0.0))
        service_time = row.get('Service_Time', 25.0)  # Default service time
        
        # Explode customer into pseudo-customers
        pseudo_customers = explode_customer(
            customer_id=row['Customer_ID'],
            demands=demands,
            location=location,
            service_time=service_time
        )
        
        # Convert pseudo-customers to DataFrame rows
        for pseudo in pseudo_customers:
            pseudo_row = {
                'Customer_ID': pseudo.customer_id,
                'Origin_ID': pseudo.origin_id,
                'Subset': '|'.join(pseudo.subset),  # Store as pipe-separated string
                'Latitude': pseudo.location[0],
                'Longitude': pseudo.location[1],
                'Service_Time': pseudo.service_time
            }
            
            # Add demand columns
            for good in goods:
                col_name = f'{good.title()}_Demand'
                pseudo_row[col_name] = pseudo.demands.get(good, 0.0)
            
            pseudo_customers_data.append(pseudo_row)
    
    result_df = pd.DataFrame(pseudo_customers_data)
    logger.info(f"Created {len(result_df)} pseudo-customers from {len(customers_df)} original customers")
    
    return result_df


def is_pseudo_customer(customer_id: str) -> bool:
    """Check if a customer ID represents a pseudo-customer (contains '::')."""
    return '::' in customer_id


def get_origin_id(customer_id: str) -> str:
    """Extract the original customer ID from a pseudo-customer ID."""
    if is_pseudo_customer(customer_id):
        return customer_id.split('::')[0]
    return customer_id


def get_subset_from_id(customer_id: str) -> Tuple[str, ...]:
    """Extract the goods subset from a pseudo-customer ID."""
    if is_pseudo_customer(customer_id):
        subset_str = customer_id.split('::')[1]
        return tuple(subset_str.split('-'))
    # For regular customers, assume they serve all their positive-demand goods
    return tuple()


# TODO: this file can be prettier.