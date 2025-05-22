"""Parser for MCVRP instance files."""

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
from ..models import MCVRPInstance


def parse_mcvrp(path: Union[str, Path]) -> MCVRPInstance:
    """Parse an MCVRP .dat file at the given path into an MCVRPInstance."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MCVRP instance file not found: {path}")

    # Read all lines, stripping whitespace
    lines = [line.strip() for line in p.open(encoding='latin-1')]

    # Parse header key:value lines
    headers: Dict[str, str] = {}
    i = 0
    while i < len(lines) and ':' in lines[i]:
        key, value = lines[i].split(':', 1)
        headers[key.strip()] = value.strip()
        i += 1

    # Check mandatory headers
    required = ['TYPE', 'DIMENSION', 'CAPACITY', 'VEHICLES', 'PRODUCT TYPES', 'COMPARTMENTS']
    for key in required:
        if key not in headers:
            raise ValueError(f"Missing required header: {key}")

    # Enforce product types and compartments
    if headers['PRODUCT TYPES'] != '3':
        raise ValueError(f"Expected 3 product types, got {headers['PRODUCT TYPES']}")
    if headers['COMPARTMENTS'] != '3':
        raise ValueError(f"Expected 3 compartments, got {headers['COMPARTMENTS']}")

    # Convert header values
    dimension = int(headers['DIMENSION'])
    capacity = int(headers['CAPACITY'])
    vehicles = int(headers['VEHICLES'])

    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, Tuple[int, int, int]] = {}
    depot_id: Optional[int] = None

    # Locate NODE_COORD_SECTION
    while i < len(lines) and lines[i] != 'NODE_COORD_SECTION':
        i += 1
    if i >= len(lines):
        raise ValueError("Missing NODE_COORD_SECTION")
    i += 1

    # Read coordinates until DEMAND_SECTION
    while i < len(lines) and lines[i] != 'DEMAND_SECTION':
        if lines[i]:
            parts = lines[i].split()
            idx = int(parts[0])
            coords[idx] = (float(parts[1]), float(parts[2]))
        i += 1
    if i >= len(lines) or lines[i] != 'DEMAND_SECTION':
        raise ValueError("Missing DEMAND_SECTION")
    i += 1

    # Read demands until DEPOT_SECTION
    while i < len(lines) and lines[i] != 'DEPOT_SECTION':
        if lines[i]:
            parts = lines[i].split()
            idx = int(parts[0])
            demands[idx] = (int(parts[1]), int(parts[2]), int(parts[3]))
        i += 1
    if i >= len(lines) or lines[i] != 'DEPOT_SECTION':
        raise ValueError("Missing DEPOT_SECTION")
    i += 1

    # Read depot ID (first non-EOF line)
    while i < len(lines):
        if lines[i] and lines[i] != 'EOF':
            depot_id = int(lines[i])
            break
        i += 1
    if depot_id is None:
        raise ValueError("Missing depot id in DEPOT_SECTION")

    # Sanity checks on counts
    if len(coords) != dimension:
        raise ValueError(f"Number of coords ({len(coords)}) does not match dimension ({dimension})")
    if len(demands) != dimension:
        raise ValueError(f"Number of demands ({len(demands)}) does not match dimension ({dimension})")

    return MCVRPInstance(
        name=p.stem,
        source_file=p,
        dimension=dimension,
        capacity=capacity,
        vehicles=vehicles,
        depot_id=depot_id,
        coords=coords,
        demands=demands
    ) 