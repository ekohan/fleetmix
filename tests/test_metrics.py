"""
Unit tests for fleet_composition metrics.
"""

from fleetmix.experiments.fleet_composition.metrics import cost_per_drop, cost_per_kg, split_rate, average_visits_per_customer
from fleetmix.core_types import FleetmixSolution, Cluster

def test_cost_per_drop():
    assert cost_per_drop(1000.0, 10) == 100.0
    assert cost_per_drop(0.0, 10) == 0.0
    assert cost_per_drop(1000.0, 0) == 0.0

def test_cost_per_kg():
    assert cost_per_kg(1000.0, 500.0) == 2.0
    assert cost_per_kg(0.0, 500.0) == 0.0
    assert cost_per_kg(1000.0, 0.0) == 0.0

def test_split_rate_no_splits():
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1'], {}, 0, 0, [], 0.0),
        Cluster(2, 'V2', 'vehicle2', ['C2'], {}, 0, 0, [], 0.0)
    ])
    assert split_rate(solution) == 0.0

def test_split_rate_with_splits():
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1::Dry'], {}, 0, 0, [], 0.0),
        Cluster(2, 'V2', 'vehicle2', ['C1::Chilled'], {}, 0, 0, [], 0.0),
        Cluster(3, 'V3', 'vehicle3', ['C2'], {}, 0, 0, [], 0.0)
    ])
    assert split_rate(solution) == 0.5  # 1 out of 2 physical customers split

def test_split_rate_empty():
    solution = FleetmixSolution(selected_clusters=[])
    assert split_rate(solution) == 0.0

def test_average_visits_per_customer_no_splits():
    """Test average visits when no customers are split."""
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1'], {}, 0, 0, [], 0.0),
        Cluster(2, 'V2', 'vehicle2', ['C2'], {}, 0, 0, [], 0.0),
        Cluster(3, 'V3', 'vehicle3', ['C3'], {}, 0, 0, [], 0.0)
    ])
    assert average_visits_per_customer(solution) == 1.0  # Each customer visited once

def test_average_visits_per_customer_with_splits():
    """Test average visits when some customers are split."""
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1::Dry'], {}, 0, 0, [], 0.0),
        Cluster(2, 'V2', 'vehicle2', ['C1::Chilled'], {}, 0, 0, [], 0.0),
        Cluster(3, 'V3', 'vehicle3', ['C2'], {}, 0, 0, [], 0.0)
    ])
    # Customer C1 visited 2 times, C2 visited 1 time
    # Average = (2 + 1) / 2 = 1.5
    assert average_visits_per_customer(solution) == 1.5

def test_average_visits_per_customer_complex_splits():
    """Test average visits with complex split patterns."""
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1::Dry', 'C1::Chilled'], {}, 0, 0, [], 0.0),  # C1 visited 2 times by V1
        Cluster(2, 'V2', 'vehicle2', ['C1::Frozen'], {}, 0, 0, [], 0.0),              # C1 visited 1 more time by V2
        Cluster(3, 'V3', 'vehicle3', ['C2', 'C3'], {}, 0, 0, [], 0.0)                # C2 and C3 each visited once by V3
    ])
    # With per-cluster deduplication by origin:
    # Cluster 1 contributes 1 visit for C1 (not 2), Cluster 2 contributes 1 for C1,
    # Cluster 3 contributes 1 each for C2 and C3 → visits: C1=2, C2=1, C3=1
    # Average = (2 + 1 + 1) / 3 = 4/3 ≈ 1.3333
    assert abs(average_visits_per_customer(solution) - 4/3) < 1e-10

def test_average_visits_per_customer_empty():
    """Test average visits with empty solution."""
    solution = FleetmixSolution(selected_clusters=[])
    assert average_visits_per_customer(solution) == 0.0

def test_average_visits_per_customer_single_cluster():
    """Test average visits with single cluster containing multiple customers."""
    solution = FleetmixSolution(selected_clusters=[
        Cluster(1, 'V1', 'vehicle1', ['C1', 'C2', 'C3'], {}, 0, 0, [], 0.0)
    ])
    assert average_visits_per_customer(solution) == 1.0  # Each customer visited once 