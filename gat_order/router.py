"""Route a board with the toolkit's negotiated router in a given net order.

The router (``m3d.negotiated.Negotiated``, PathFinder-style) connects every net
with Dijkstra on congestion-aware costs and negotiates overused vertices away
over up to ``max_iters`` passes. The order nets are routed in is the only thing
changed here: a subclass hands it the order instead of its default
(largest bounding box first, which is how the hard tier's baseline was made).
The order changes the result: on hard case_01 random orders scored between
0.96 and 1.07, and some orders never become legal.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from m3d.checker import check
from m3d.model import Instance, Submission
from m3d.negotiated import Negotiated


class OrderedNegotiated(Negotiated):
    def __init__(self, inst: Instance, order: List[int], **kw):
        super().__init__(inst, **kw)
        self.given_order = list(order)

    def _order_nets(self) -> List[int]:
        return list(self.given_order)


def default_order(inst: Instance) -> List[int]:
    """The router's own order (largest bounding box first): the baseline's."""
    return Negotiated(inst)._order_nets()


def route(inst: Instance, order: List[int], max_iters: int = 50
          ) -> Tuple[Optional[Submission], Optional[int], float]:
    """(submission, total delay, seconds); submission and delay are None if the
    routing never became legal. ``order`` lists net ids."""
    t0 = time.time()
    sub, _stats = OrderedNegotiated(inst, order, max_iters=max_iters).run()
    if sub is None:
        return None, None, time.time() - t0
    res = check(inst, sub)
    if not res.legal:
        return None, None, time.time() - t0
    return sub, res.total_delay, time.time() - t0
