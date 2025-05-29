"""
common.py

Common data structures used across the clustering process.
"""

from typing import Dict, List
from dataclasses import dataclass, field

from fleetmix.utils.logging import FleetmixLogger
logger = FleetmixLogger.get_logger(__name__)

class Symbols:
    """Unicode symbols for logging."""
    CHECKMARK = "✓"
    CROSS = "✗"

@dataclass
class Cluster:
    """Represents a cluster of customers that can be served by a vehicle configuration."""
    cluster_id: int
    config_id: int
    customers: List[str]
    total_demand: Dict[str, float]
    centroid_latitude: float
    centroid_longitude: float
    goods_in_config: List[str]
    route_time: float
    method: str = ''
    tsp_sequence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert cluster to dictionary format."""
        data = {
            'Cluster_ID': self.cluster_id,
            'Config_ID': self.config_id,
            'Customers': self.customers,
            'Total_Demand': self.total_demand,
            'Centroid_Latitude': self.centroid_latitude,
            'Centroid_Longitude': self.centroid_longitude,
            'Goods_In_Config': self.goods_in_config,
            'Route_Time': self.route_time,
            'Method': self.method
        }
        # Only add sequence if it exists
        if self.tsp_sequence:
            data['TSP_Sequence'] = self.tsp_sequence
        return data

@dataclass
class ClusteringSettings:
    """Encapsulates all settings required for a clustering run."""
    method: str
    goods: List[str]
    depot: Dict[str, float]
    avg_speed: float
    service_time: float
    max_route_time: float
    max_depth: int
    route_time_estimation: str
    geo_weight: float
    demand_weight: float 