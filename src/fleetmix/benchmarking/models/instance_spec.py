from dataclasses import dataclass
from typing import Dict, List

from fleetmix.core_types import DepotLocation, VehicleSpec


@dataclass(slots=True, frozen=True)
class InstanceSpec:
    """Specification for a VRP benchmark instance.

    This dataclass captures the essential structure of a VRP instance
    including expected vehicle count, depot location, goods present,
    and available vehicle types with their specifications.
    """

    expected_vehicles: int
    depot: DepotLocation
    goods: List[str]  # present goods
    vehicles: Dict[str, VehicleSpec]  # key = vehicle type
