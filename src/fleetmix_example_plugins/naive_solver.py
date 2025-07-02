"""Naive solver adapter plugin for FleetMix.

Demonstrates how to plug in a *custom* PuLP solver adapter via the FleetMix
registry.  This adapter simply forwards to CBC but is registered under the key
``naive`` so it can be activated with::

    export FSM_SOLVER=naive

(or programmatically ``os.environ['FSM_SOLVER'] = 'naive'`` **before** the first
FleetMix import).
"""

from __future__ import annotations

import pulp

from fleetmix.registry import register_solver_adapter

# Override the default CBC adapter with a demo variant ----------------------------------
# Registering under the *same* key ('cbc') makes `FSM_SOLVER=cbc` pick this adapter.
# ------------------------------------------------------------------------------------


@register_solver_adapter("cbc")
class RelaxedCbcAdapter:
    """Thin wrapper around PuLP's CBC with relaxed settings for speed."""

    def get_pulp_solver(self, *, verbose: bool = False, gap_rel: float | None = 0):  # noqa: D401,E501
        msg = 1 if verbose else 0
        # Relaxed relative gap to speed up demo runs
        gap = 0.2 if gap_rel is None else gap_rel  # 20 % default for demo
        return pulp.PULP_CBC_CMD(msg=msg, gapRel=gap)

    @property
    def name(self) -> str:  # noqa: D401
        return "Relaxed CBC"

    @property
    def available(self) -> bool:  # noqa: D401
        return True
