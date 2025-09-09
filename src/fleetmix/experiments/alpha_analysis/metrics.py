"""
Normalized metrics for alpha analysis.
"""

from collections import defaultdict
from typing import Dict

from fleetmix.core_types import FleetmixSolution


def _count_customer_visits(solution: FleetmixSolution) -> Dict[str, int]:
    """Helper function to count visits per physical customer across all clusters."""
    customer_counts: Dict[str, int] = defaultdict(int)
    for cluster in solution.selected_clusters:
        # Count each physical customer at most once per cluster visit
        origins_in_cluster = set()
        for cust_id in cluster.customers:
            origin_id = cust_id.split("::")[0] if "::" in cust_id else cust_id
            origins_in_cluster.add(origin_id)
        for origin_id in origins_in_cluster:
            customer_counts[origin_id] += 1
    return dict(customer_counts)


def cost_per_drop(total_cost: float, num_customers: int) -> float:
    """Cost per customer drop."""
    if num_customers == 0:
        return 0.0
    return total_cost / num_customers


def cost_per_kg(total_cost: float, total_demand_kg: float) -> float:
    """Cost per kg of demand."""
    if total_demand_kg == 0:
        return 0.0
    return total_cost / total_demand_kg


def split_rate(solution: FleetmixSolution) -> float:
    """Fraction of physical customers served by multiple vehicles (split stops)."""
    customer_counts = _count_customer_visits(solution)
    if not customer_counts:
        return 0.0
    num_split = sum(1 for count in customer_counts.values() if count > 1)
    total_physical = len(customer_counts)
    return num_split / total_physical


def average_visits_per_customer(solution: FleetmixSolution) -> float:
    """Average number of vehicle visits (stops) per physical customer."""
    customer_counts = _count_customer_visits(solution)
    if not customer_counts:
        return 0.0
    return sum(customer_counts.values()) / len(customer_counts)
