"""Models for benchmarking functionality."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class MCVRPInstance:
    """Container for parsed MCVRP instance data."""
    name: str
    source_file: Path
    dimension: int
    capacity: int
    vehicles: int
    depot_id: int
    coords: Dict[int, Tuple[float, float]]
    demands: Dict[int, Tuple[int, int, int]]

    def customers(self) -> List[int]:
        """Return all customer node IDs (excluding the depot)."""
        return [node_id for node_id in self.coords.keys() if node_id != self.depot_id]

@dataclass
class CVRPInstance:
    """Container for parsed CVRP instance data."""
    name: str
    dimension: int
    capacity: int
    depot_id: int
    coordinates: Dict[int, Tuple[float, float]]
    demands: Dict[int, float]
    edge_weight_type: str
    num_vehicles: int

@dataclass
class CVRPSolution:
    """Container for CVRP solution data."""
    routes: List[List[int]]
    cost: float
    num_vehicles: int
    expected_vehicles: int 