"""One challenge case loaded into the C++ router core."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from routing.paths import CHALLENGE_ROOT  # noqa: F401  (puts the challenge checkout on sys.path)
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, NetRoute, Submission  # noqa: E402

from core import _core  # noqa: E402
from routing.modes import make_modes, negotiation_mode  # noqa: E402

Box = Tuple[int, int, int, int]


class Problem:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.W, self.H, self.L = inst.width, inst.height, inst.layers
        self.WH = self.W * self.H
        self.layer_delay = list(inst.layer_delay)
        pin_v = {p.id: self.vid(p.vertex()) for p in inst.pins}
        # nets are addressed by their index in inst.nets
        self.net_ids = [n.id for n in inst.nets]
        self.drivers = [pin_v[n.driver] for n in inst.nets]
        self.sinks = [[pin_v[s] for s in n.sinks] for n in inst.nets]
        pin_vids, pin_nets = [], []
        for i, n in enumerate(inst.nets):
            for p in n.pins():
                pin_vids.append(pin_v[p]); pin_nets.append(i)
        self.router = _core.Router(self.W, self.H, self.L, self.layer_delay, inst.via_delay,
                                   pin_vids, pin_nets, self.drivers, self.sinks)
        self.modes = make_modes(self.layer_delay)
        self.N = len(inst.nets)
        self.ideal = [self.router.ideal_delay(i) for i in range(self.N)]
        if min(self.ideal) < 0:
            raise RuntimeError(f"{inst.name}: a net is unroutable even on an empty grid")
        self.lower_bound = sum(self.ideal)
        self.pin_bbox = [self._bbox([self.drivers[i]] + self.sinks[i]) for i in range(self.N)]
        top = self.L - 1
        # static per-net facts used by the network's net features
        self.n_sinks = np.array([len(s) for s in self.sinks], dtype=np.float32)
        self.driver_top = np.array([float(self.coord(d)[2] == top) for d in self.drivers], dtype=np.float32)
        self.sinks_top = np.array([np.mean([self.coord(v)[2] == top for v in s]) for s in self.sinks], dtype=np.float32)
        self.cross_die = np.array([float(any(self.coord(v)[2] != self.coord(d)[2] for v in s))
                                   for d, s in zip(self.drivers, self.sinks)], dtype=np.float32)
        self.pin_vids = [[self.drivers[i]] + self.sinks[i] for i in range(self.N)]

    @staticmethod
    def load(path: str) -> "Problem":
        return Problem(Instance.load(path))

    # -- geometry ---------------------------------------------------------------
    def vid(self, v) -> int:
        x, y, z = v
        return (z * self.H + y) * self.W + x

    def coord(self, vid: int) -> Tuple[int, int, int]:
        z, r = divmod(vid, self.WH)
        y, x = divmod(r, self.W)
        return (x, y, z)

    def _bbox(self, vids: List[int]) -> Box:
        xs, ys = [], []
        for v in vids:
            x, y, _ = self.coord(v); xs.append(x); ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    def grow(self, box: Box, margin: int) -> Box:
        x0, y0, x1, y1 = box
        return (max(0, x0 - margin), max(0, y0 - margin), min(self.W - 1, x1 + margin), min(self.H - 1, y1 + margin))

    @staticmethod
    def union(a: Box, b: Box) -> Box:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def bbox_semi(self, i: int) -> int:
        x0, y0, x1, y1 = self.pin_bbox[i]
        return (x1 - x0) + (y1 - y0)

    # -- initial routing --------------------------------------------------------
    SCHEDULES = [(0.3, 1.2, 0.3), (0.5, 1.3, 0.5), (0.5, 1.5, 0.5)]  # (pres0, pres_mult, hist_fac)

    def initial_route(self, schedules=None) -> bool:
        """Negotiated congestion over two net orders (largest box first, and the reverse) and
        several pressure schedules; keep the best legal result."""
        by_bbox = sorted(range(self.N), key=lambda i: (-self.bbox_semi(i), i))
        orders = [by_bbox, list(reversed(by_bbox))]
        m = negotiation_mode(self.layer_delay)
        best, best_routes = None, None
        for sched in schedules or self.SCHEDULES:
            for order in orders:
                if self.router.negotiate(order, m, 300, *sched):
                    d = self.router.total_delay()
                    if best is None or d < best:
                        best, best_routes = d, self.snapshot()
        if best_routes is None:
            return False
        self.restore(best_routes)
        return True

    def snapshot(self) -> List[List[Tuple[int, int]]]:
        return [self.router.route_edges(i) for i in range(self.N)]

    def restore(self, snap) -> None:
        """Reinstall routes exactly from a snapshot of tree edges."""
        r = self.router
        for i in range(self.N):
            r.rip_up(i)
        for i, edges in enumerate(snap):
            r.set_route(i, edges)
        r.commit()

    # -- export -----------------------------------------------------------------
    def to_submission(self) -> Submission:
        routes = []
        for i in range(self.N):
            edges = [(self.coord(a), self.coord(b)) for a, b in self.router.route_edges(i)]
            routes.append(NetRoute(net=self.net_ids[i], edges=edges))
        return Submission(instance=self.inst.name, routes=routes)

    def verify(self):
        return check(self.inst, self.to_submission())
