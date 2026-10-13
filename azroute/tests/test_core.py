"""Regression tests: legality and exact delay vs the independent m3d checker, undo/rollback,
exact replay of logged reroutes (including network-chosen price canvases), and the network's
input/augmentation plumbing.   python -m unittest azroute.tests.test_core"""
import random
import unittest

import numpy as np

from azroute.lns import Improver, greedy_until_stall, replay
from azroute.problem import Problem
from azroute.search import Search, completion_evaluator, drive
from azroute.tiers import cases


def _case(tier, i):
    return Problem.load(cases(tier)[i][1])


class CoreTests(unittest.TestCase):
    def test_initial_routing_is_legal_and_exact(self):
        for tier, i in [("intro", 4), ("hard", 0)]:
            P = _case(tier, i)
            self.assertTrue(P.initial_route())
            res = P.verify()
            self.assertTrue(res.legal, res.reasons)
            self.assertEqual(res.total_delay, P.router.total_delay())
            self.assertGreaterEqual(P.router.total_delay(), P.lower_bound)

    def test_rollback_restores_routes(self):
        P = _case("hard", 0)
        P.initial_route()
        before = P.snapshot()
        r = P.router
        cp = r.checkpoint()
        for n in range(10):
            r.rip_up(n)
        r.route_net(3, P.modes["DELAY"], P.full_box())
        r.rollback(cp)
        self.assertEqual(P.snapshot(), before)

    def test_improvement_stays_legal_and_replays_exactly(self):
        import torch
        from azroute.model import RouteNet
        from azroute.worker import NetEvaluator
        P = _case("hard", 0)
        P.initial_route()
        imp = Improver(P, random.Random(0), max_nets=5)
        greedy_until_stall(imp, stall_tries=50, max_seconds=5)
        net = NetEvaluator("", torch.device("cpu"))
        net.model = RouteNet.preset("tiny").eval()
        search = Search(imp, n_full=8, n_fast=4, p_full=0.5)
        records = []
        for k in range(4):
            evaluate = completion_evaluator(imp) if k % 2 else (lambda st: net([st])[0])
            *_, recs = drive(search.reroute(imp.neighbourhood(imp.pick_seed())), evaluate)
            records += recs
        res = P.verify()
        self.assertTrue(res.legal, res.reasons)
        self.assertEqual(res.total_delay, P.router.total_delay())
        Q = _case("hard", 0)
        Q.initial_route()
        replay(Q, imp.log)
        self.assertEqual(Q.snapshot(), P.snapshot())
        self.assertTrue(records)
        for r in records:
            self.assertAlmostEqual(float(r["pi"].astype(np.float32).sum()), 1.0, places=2)

    def test_augmentation_is_a_group_action(self):
        import torch
        from azroute.learner import ReplayBuffer, augment, to_device
        P = _case("hard", 0)
        P.initial_route()
        imp = Improver(P, random.Random(1), max_nets=4)
        search = Search(imp, n_full=4, n_fast=2, p_full=1.0)
        recs = []
        while not recs:
            *_, recs = drive(search.reroute(imp.neighbourhood(imp.pick_seed())), completion_evaluator(imp))
        buf = ReplayBuffer(16, P.L)
        buf.add({k: np.stack([r[k] for r in recs]) for k in recs[0]})
        b = to_device(buf.sample(4, np.random.default_rng(0)), torch.device("cpu"))
        for sym in (1, 2, 8):   # flips are involutions
            twice = augment(augment(b, sym), sym)
            for k in ("maps", "rois", "nodes", "ld"):
                self.assertTrue(torch.allclose(twice[k].float(), b[k].float(), atol=1e-6), (sym, k))


if __name__ == "__main__":
    unittest.main()
