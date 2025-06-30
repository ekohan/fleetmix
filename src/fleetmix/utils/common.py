"""Common one-liner helpers shared across FleetMix modules."""

from __future__ import annotations

from typing import Any

from fleetmix.core_types import FleetmixSolution

__all__ = [
    "to_cfg_key",
    "build_zero_filled_demands",
    "baseline_is_valid",
]


def to_cfg_key(cfg_id: str | int) -> str:
    """Return configuration key as *string* – canonical representation throughout the codebase."""
    return str(cfg_id)


def build_zero_filled_demands(all_goods: list[str]) -> dict[str, float]:
    """Return a dict with all goods initialised to 0.0 demand."""
    return {g: 0.0 for g in all_goods}


def baseline_is_valid(sol: FleetmixSolution) -> bool:
    """A baseline solution is usable when it serves every customer with at least one vehicle."""
    return sol.total_vehicles > 0 and not sol.missing_customers
