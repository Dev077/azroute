"""Export the current routings as checked submissions.

    python -m pipeline.export --work /content/work --out /content/drive/MyDrive/azroute/export

For every case with a stored routing: rebuild it, check it with the independent m3d
checker, and write <out>/submissions/<tier>/<case>.sol.json. Also writes runtime.json per
tier and summary.md (aggregate vs the suite baseline per tier). A case's runtime is the worker
seconds spent on it plus a share of the learner's time proportional to the case's training records.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

from routing.problem import Problem
from routing.tiers import all_cases
from pipeline.worker import load_routing


def export(work: str, out: str) -> str:
    stats_path = os.path.join(work, "case_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    overhead = stats.get("_meta", {}).get("learner_seconds", 0.0)
    total_records = sum(s.get("records", 0) for k, s in stats.items() if not k.startswith("_")) or 1
    rows = defaultdict(list)
    runtimes = defaultdict(dict)
    for tier, name, path, base in all_cases():
        key = f"{tier}/{name}"
        stored = load_routing(work, key)
        if stored is None:
            continue
        P = Problem.load(path)
        P.restore(stored[1])
        res = P.verify()
        if not res.legal:
            print(f"{key}: NOT LEGAL ({res.reasons[:3]}) - skipped")
            continue
        d = os.path.join(out, "submissions", tier)
        os.makedirs(d, exist_ok=True)
        P.to_submission().save(os.path.join(d, f"{name}.sol.json"))
        s = stats.get(key, {})
        secs = s.get("seconds", 0.0) + overhead * s.get("records", 0) / total_records
        runtimes[tier][name] = round(secs, 2)
        rows[tier].append((name, base, res.total_delay, P.lower_bound, s.get("initial_delay")))
    lines = ["# azroute export", "", "| tier | cases | aggregate vs baseline | total delay | baseline total | delay / lower bound |",
             "|---|---:|---:|---:|---:|---:|"]
    for tier, rs in rows.items():
        gm = math.exp(sum(math.log(b / d) for _, b, d, _, _ in rs) / len(rs))
        tot, btot, lb = sum(r[2] for r in rs), sum(r[1] for r in rs), sum(r[3] for r in rs)
        lines.append(f"| {tier} | {len(rs)} | {gm:.4f} | {tot} | {btot} | {tot / lb:.4f} |")
        json.dump(runtimes[tier], open(os.path.join(out, "submissions", tier, "runtime.json"), "w"), indent=1)
    lines += ["", "Per case:", "", "| case | baseline | initial | final | final / lower bound |", "|---|---:|---:|---:|---:|"]
    for tier, rs in rows.items():
        for name, b, d, lb, init in rs:
            lines.append(f"| {tier}/{name} | {b} | {init if init is not None else '-'} | {d} | {d / lb:.4f} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "summary.md"), "w").write(text)
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.work, a.out)
