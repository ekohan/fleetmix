"""
Preprocessing utilities for FleetMix.


TODO: this is for split-stop. Maybe rename or bring more preprocessing here.
"""

from .demand import explode_customer, maybe_explode

__all__ = ['explode_customer', 'maybe_explode'] 