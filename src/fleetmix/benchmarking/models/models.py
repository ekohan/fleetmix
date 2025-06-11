"""Models for benchmarking functionality."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MCVRPInstance:
    """Container for parsed MCVRP instance data."""

    name: str
    source_file: Path
    dimension: int
    capacity: int
    vehicles: int
    depot_id: int
    coords: dict[int, tuple[float, float]]
    demands: dict[int, tuple[int, int, int]]

    def customers(self) -> list[int]:
        """Return all customer node IDs (excluding the depot)."""
        return [node_id for node_id in self.coords.keys() if node_id != self.depot_id]


@dataclass
class CVRPInstance:
    """Container for parsed CVRP instance data."""

    name: str
    dimension: int
    capacity: int
    depot_id: int
    coordinates: dict[int, tuple[float, float]]
    demands: dict[int, float]
    edge_weight_type: str
    num_vehicles: int


@dataclass
class CVRPSolution:
    """Container for CVRP solution data."""

    routes: list[list[int]]
    cost: float
    num_vehicles: int
