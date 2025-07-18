"""Naive solver adapter plugin for FleetMix.

Demonstrates how to plug in a *custom* PuLP solver adapter via the FleetMix
registry.  This adapter simply forwards to CBC but is registered under the key
``naive``.

This file is *only* for demonstration purposes, it shows how a user can add
custom components without modifying FleetMix's source code.
See examples/custom_solver_adapter.py.
"""

from __future__ import annotations

import pulp

from fleetmix.config.params import RuntimeParams
from fleetmix.registry import register_solver_adapter


@register_solver_adapter("naive")
class RelaxedCbcAdapter:
    """Thin wrapper around PuLP's CBC with relaxed settings for speed."""

    def get_pulp_solver(self, params: RuntimeParams) -> pulp.LpSolver:  # noqa: D401,E501
        msg = 1 if params.verbose else 0
        # Relaxed relative gap to speed up demo runs
        gap = 0.2 if params.gap_rel is None else params.gap_rel  # 20 % default for demo
        kwargs = {"msg": msg, "gapRel": gap}

        if params.time_limit is not None and params.time_limit > 0:
            kwargs["timeLimit"] = params.time_limit

        return pulp.PULP_CBC_CMD(**kwargs)

    @property
    def name(self) -> str:  # noqa: D401
        return "Relaxed CBC"

    @property
    def available(self) -> bool:  # noqa: D401
        return True
