"""Regression tests: legality and exact delay vs the independent m3d checker, undo/rollback,
exact replay of logged reroutes (including network-chosen price canvases), and the network's
input/augmentation plumbing.   python -m unittest tests.test_core"""
import random
import unittest

import numpy as np

from routing.lns import Improver, greedy_until_stall, replay
from routing.problem import Problem
from search.search import Search, completion_evaluator, drive
from routing.tiers import cases


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
        from network.model import RouteNet
        from pipeline.worker import NetEvaluator
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
        from network.learner import ReplayBuffer, augment, to_device
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
        self.assertGreater(buf.sample(64, np.random.default_rng(1), prio=0.5)["y"].shape[0], 0)

    def test_deferred_search_matches_sequential(self):
        """Batching a halving round into one request must not change any decision or record."""
        import torch
        from network.model import RouteNet
        from pipeline.worker import NetEvaluator
        torch.manual_seed(0)
        net = NetEvaluator("", torch.device("cpu"))
        net.model = RouteNet.preset("tiny").eval()
        runs = []
        for defer in (False, True):
            P = _case("hard", 0)
            P.initial_route()
            imp = Improver(P, random.Random(3), max_nets=5)
            search = Search(imp, n_full=16, n_fast=4, p_full=0.5, defer=defer, rng=random.Random(7))
            recs = []
            for _ in range(3):
                *_, r = drive(search.reroute(imp.neighbourhood(imp.pick_seed())), lambda x: net([x])[0])
                recs += r
            runs.append((P.snapshot(), recs))
        (snap_a, recs_a), (snap_b, recs_b) = runs
        self.assertEqual(snap_a, snap_b)
        self.assertEqual(len(recs_a), len(recs_b))
        for ra, rb in zip(recs_a, recs_b):
            np.testing.assert_allclose(ra["pi"].astype(np.float32), rb["pi"].astype(np.float32), atol=1e-3)
            self.assertAlmostEqual(float(ra["spread"]), float(rb["spread"]), places=4)

    def test_keeping_worse_reroutes_restores_the_best_exactly(self):
        """With worse reroutes kept, restore_best() must give back the best routing, and the
        truncated log must replay to it exactly."""
        from routing.lns import RecordToRecord, greedy_policy
        P = _case("hard", 0)
        P.initial_route()
        imp = Improver(P, random.Random(5), max_nets=6, accept=RecordToRecord(0.02, 1e6))
        worse = 0
        for _ in range(300):
            before = imp.total
            imp.reroute(imp.neighbourhood(imp.pick_seed()), greedy_policy)
            self.assertEqual(imp.total, P.router.total_delay())
            worse += imp.total > before
        self.assertGreater(worse, 0)
        imp.restore_best()
        res = P.verify()
        self.assertTrue(res.legal, res.reasons)
        self.assertEqual(res.total_delay, imp.best_total)
        Q = _case("hard", 0)
        Q.initial_route()
        replay(Q, imp.log)
        # a restored route may list its edges in another order; compare edge sets
        self.assertEqual([sorted(e) for e in Q.snapshot()], [sorted(e) for e in P.snapshot()])


if __name__ == "__main__":
    unittest.main()
