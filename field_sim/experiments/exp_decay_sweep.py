"""Decay-rate sensitivity of the proposed method under Case Study 1's FREE
(live, intra-round) state access - no synchronization model, so the decay
effect is isolated from any access delay.

Everything is Case Study 1 (exp_ccg_vs_fragmentation: INSTANCES, 8
Humanoid-linked agents, M1 threshold 0.5, r = 26, 80 rounds, seeds 0-7,
min confidence 0.05) except the evaporation rate: 0, 0.05, 0.15 (Case Study
1's value), 0.30. The existing decay -> reinforcement interaction is kept:
decay lowers the stored confidence, and a later observation more than 0.2
more confident than the stored value is a REINFORCE upload
(field_sim.selection). Upload classes are recorded so that interaction can
be read off directly.

Per round: coverage / precision (unique triples; precision NaN when nothing
is held), stored traces, wrong stored traces split by fact_type
(observed / inferred), and for wrong inferred traces how long their value
has stood (t - value_since) - at decay 0 nothing is ever removed, so this
measures how long stale inferences linger.

Outputs: field_sim/experiments/results_additional/decay_sweep/
Run: python -m field_sim.experiments.exp_decay_sweep --rates 0.02 --tag _p002
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from field_sim.agent import sync_agent_locations
from field_sim.evaluation import edge_triples, reference_state
from field_sim.experiments import exp_ccg_vs_fragmentation as ccg
from field_sim.experiments import exp_common as sync
from field_sim.experiments.harness import build_world
from field_sim.reasoning import FieldReasoner

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "decay_sweep")
DECAYS = [0.0, 0.05, 0.15, 0.30]
RATES = sync.RATES
SEEDS = sync.SEEDS
FINAL_WINDOW = ccg.STEADY


def run_one(rate: float, decay: float, seed: int, steps: int = ccg.STEPS) -> dict:
    cfg = sync._cfg(rate, seed, evaporation=decay)
    cfg.steps = steps
    env, kb, agents, field, _ = build_world(cfg, policy_factory=None)
    m1 = ccg._m1_policy_factory(kb)
    reasoner = FieldReasoner()
    stream = hashlib.sha256()
    rows = []
    for t in range(1, steps + 1):
        env.step(t)
        sync_agent_locations(agents, env)
        cand = {a.agent_id: a.form_candidate_traces(a.observe(env, kb, t), kb, t) for a in agents}
        stream.update(ccg.candidate_fingerprint(t, [c for a in agents for c in cand[a.agent_id]]).encode())
        cls = Counter()
        for a in agents:
            for c in cand[a.agent_id]:
                d = m1.evaluate(c, field, t, kb)
                if d.transmit:
                    field.set(c, t)
                    cls[d.classification] += 1
        removed = field.decay(t)
        field.reason(kb, t)
        truth = edge_triples(reference_state(env, kb, t, reasoner))
        q = sync.quality(edge_triples(field.edges), truth)
        wrong = [e for e in field.edges.values() if e.triple not in truth]
        wrong_inf = [e for e in wrong if e.fact_type == "inferred"]
        rows.append({"t": t, **q, "uploads": sum(cls.values()), **{f"cls_{k}": v for k, v in cls.items()},
                     "stored_edges": len(field.edges), "evaporated": len(removed),
                     "wrong_observed": len(wrong) - len(wrong_inf), "wrong_inferred": len(wrong_inf),
                     "wrong_inferred_age_mean": statistics.mean(t - e.value_since for e in wrong_inf) if wrong_inf else 0.0,
                     "wrong_inferred_age_max": max((t - e.value_since for e in wrong_inf), default=0),
                     "inferred_held": sum(1 for e in field.edges.values() if e.fact_type == "inferred")})
    return {"rounds": rows, "stream_sha": stream.hexdigest()}


def _job(args):
    t0 = time.time()
    out = run_one(*args)
    out["seconds"] = time.time() - t0
    return args, out


def seed_rows(runs):
    out = []
    for (rate, decay, seed), o in sorted(runs.items()):
        R, last = o["rounds"], o["rounds"][-FINAL_WINDOW:]
        prec = [r["precision"] for r in last if not math.isnan(r["precision"])]
        cls = Counter()
        for r in R:
            cls.update({k: v for k, v in r.items() if k.startswith("cls_")})
        out.append({"rate": RATES[rate], "change_prob": rate, "decay": decay, "seed": seed,
                    "uploads": sum(r["uploads"] for r in R), "uploads_round1": R[0]["uploads"],
                    **{k: cls.get(k, 0) for k in ("cls_novel", "cls_reinforce", "cls_refresh", "cls_correction", "cls_redundant")},
                    "coverage_last20": statistics.mean(r["coverage"] for r in last),
                    "precision_last20": statistics.mean(prec) if prec else math.nan,
                    "precision_nan_rounds_last20": FINAL_WINDOW - len(prec),
                    **{f"{m}_last20": statistics.mean(r[m] for r in last)
                       for m in ("stored_edges", "held", "matched", "wrong_observed", "wrong_inferred", "inferred_held",
                                 "wrong_inferred_age_mean")},
                    "wrong_inferred_age_max_last20": max(r["wrong_inferred_age_max"] for r in last),
                    "seconds": round(o["seconds"], 1)})
    return out


def condition_rows(srows):
    by = defaultdict(list)
    for r in srows:
        by[(r["rate"], r["decay"])].append(r)
    out = []
    for (rate, decay), rs in sorted(by.items()):
        row = {"rate": rate, "decay": decay, "seeds": len(rs)}
        for m in [k for k in rs[0] if k not in ("rate", "change_prob", "decay", "seed")]:
            row[f"{m}_mean"], row[f"{m}_sd"] = sync._ms([r[m] for r in rs])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"config{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"access": "free (Case Study 1 live intra-round)", "decays": DECAYS, "min_confidence": 0.05,
                   "change_probs": args.rates, "radius": sync.RADIUS, "seeds": SEEDS, "steps": ccg.STEPS,
                   "policy": "M1 threshold 0.5", "upload_budget": None}, f, indent=2)
    before = sync.manifest(sync.PRESERVED)
    jobs = [(r, d, s) for r in args.rates for d in DECAYS for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        runs = dict(ex.map(_job, jobs))
    rw = [{"rate": RATES[r], "decay": d, "seed": s, **row} for (r, d, s), o in sorted(runs.items()) for row in o["rounds"]]
    sync._write_csv(os.path.join(OUT_DIR, f"roundwise{args.tag}.csv"), rw)
    srows = seed_rows(runs)
    sync._write_csv(os.path.join(OUT_DIR, f"seed_summary{args.tag}.csv"), srows)
    crows = condition_rows(srows)
    sync._write_csv(os.path.join(OUT_DIR, f"condition_summary{args.tag}.csv"), crows)
    checks = {"candidate_stream_identical_across_decay": all(
        len({runs[(r, d, s)]["stream_sha"] for d in DECAYS}) == 1 for r in args.rates for s in SEEDS),
        "existing_results_unchanged": sync.manifest(sync.PRESERVED) == before}
    with open(os.path.join(OUT_DIR, f"summary{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"validation": checks, "elapsed_seconds": time.time() - t0}, f, indent=2)
    print("validation:", checks)
    for r in crows:
        print(f"{r['rate']:8s} decay={r['decay']:.2f} up {r['uploads_mean']:6.0f}±{r['uploads_sd']:4.0f}"
              f" N/RI/RF/C {r['cls_novel_mean']:.0f}/{r['cls_reinforce_mean']:.0f}/{r['cls_refresh_mean']:.0f}/{r['cls_correction_mean']:.0f}"
              f" cov {r['coverage_last20_mean']:.3f}±{r['coverage_last20_sd']:.3f} prec {r['precision_last20_mean']:.3f}±{r['precision_last20_sd']:.3f}"
              f" stored {r['stored_edges_last20_mean']:.0f} wrongObs {r['wrong_observed_last20_mean']:.1f}"
              f" wrongInf {r['wrong_inferred_last20_mean']:.1f} (age {r['wrong_inferred_age_mean_last20_mean']:.1f})")


if __name__ == "__main__":
    main()
