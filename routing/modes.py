"""Routing modes: parameter presets for the C++ tree builder.

Every mode keeps the search cost >= the true delay (multipliers >= 1, secondary
terms >= 0), so A* stays exact. Secondary terms are tiny unless a mode is meant to
trade delay for something else, so they only break ties between equal-delay trees.
"""
from __future__ import annotations

from typing import Dict, List

from . import _core

TIE_RESERVE = 1e-3   # entering a reserved pin column (pin access for not-yet-routed nets)
TIE_WIRE = 1e-5      # per new vertex: among equal-delay trees prefer the one using fewer resources

# KEEP reinstalls the net's route from before the rip-up (when its cells are still free),
# so a reroute can always reproduce the incumbent and change only some nets.
KEEP = "KEEP"
MODE_NAMES: List[str] = [KEEP, "DELAY", "SPARE_MID", "LEARNED_PRICE", "OUTER_TRUNK", "FAR_SINK_FIRST", "COMPACT"]


def _cheapest_layers(layer_delay: List[int]) -> List[int]:
    lo = min(layer_delay)
    return [z for z, d in enumerate(layer_delay) if d == lo]


def make_modes(layer_delay: List[int], price_weight: float = 2.0) -> Dict[str, "_core.ModeParams"]:
    L = len(layer_delay)
    mid = set(_cheapest_layers(layer_delay))

    def base() -> "_core.ModeParams":
        m = _core.ModeParams()
        m.layer_mult = [1.0] * L
        m.reserve_weight = TIE_RESERVE
        m.wire_weight = TIE_WIRE
        return m

    modes = {}
    m = base(); modes["DELAY"] = m
    m = base(); m.layer_mult = [1.5 if z in mid else 1.0 for z in range(L)]; modes["SPARE_MID"] = m
    m = base(); m.price_weight = price_weight; modes["LEARNED_PRICE"] = m
    m = base(); m.layer_mult = [3.0 if z in mid else 1.0 for z in range(L)]; modes["OUTER_TRUNK"] = m
    m = base(); m.far_sink_first = True; modes["FAR_SINK_FIRST"] = m
    m = base(); m.zero_seed = True; modes["COMPACT"] = m
    return modes


def negotiation_mode(layer_delay: List[int]) -> "_core.ModeParams":
    m = _core.ModeParams()
    m.layer_mult = [1.0] * len(layer_delay)
    m.wire_weight = TIE_WIRE
    return m
