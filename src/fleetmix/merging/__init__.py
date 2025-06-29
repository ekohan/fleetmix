"""Merging utilities for combining clusters.

This module provides functionality for merging clusters that is shared between
the pre-optimization cluster generation phase and the post-optimization improvement phase.
"""

from .core import generate_merge_phase_clusters

__all__ = ["generate_merge_phase_clusters"]
