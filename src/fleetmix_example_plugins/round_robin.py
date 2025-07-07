"""Example plugin module registering a trivial round-robin clusterer.

This file is *only* for demonstration purposes, it shows how a user can add
custom components without modifying FleetMix's source code.

See examples/custom_clustering.py.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from fleetmix.core_types import CapacitatedClusteringContext
from fleetmix.registry import register_clusterer


@register_clusterer("round_robin")
class RoundRobinClusterer:
    """Assigns customers to clusters in a simple round-robin fashion."""

    def fit(
        self,
        customers: pd.DataFrame,
        *,
        context: CapacitatedClusteringContext,
        n_clusters: int,
    ) -> List[int]:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        return [i % n_clusters for i in range(len(customers))]
