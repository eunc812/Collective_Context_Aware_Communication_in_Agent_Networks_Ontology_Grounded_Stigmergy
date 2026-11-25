"""Uploads versus the number of agents: proposed (M1) vs change-triggered.

Agents ARE Humanoids (the existing model: one sensor per Humanoid,
field_sim.agent.build_humanoid_linked_agents), so the number of agents N is
varied by the number of Humanoids: N = 4, 8, 16, 32. Everything else is the
Case Study 1 world and setup (exp_ccg_vs_fragmentation): 8 Pedestrians, the
same regions (4 Area, 4 Room, 4 Corridor, 2 WorkZone, 2 Passage - a fixed
4x4 grid) and background objects, r = 26, evaporation 0.15, floor 0.05,
agent_move_step 6.0, 80 rounds, seeds 0-7, M1 threshold 0.5, the field's
decay and single-pass reasoning. Change probability p = 0.02 / 0.10 / 0.35.
N = 8 is exactly Case Study 1's population (asserted), so at p = 0.02 the
proposed run must reproduce Case Study 1 (validated per seed).

Adding Humanoids also adds entities (each Humanoid has its own location,
near, motion facts), so the reference state grows with N. Worlds with
different N are independent (a different Humanoid count changes the
environment's random draws); within one (N, p, seed) both methods see the
same world and the same candidate stream (validated).

Methods
  proposed          M1Policy against the live collective state (Case Study 1).
  change_triggered  field_sim.change_triggered: an agent sends a candidate
                    only if its object differs from what THAT agent last sent
                    for that key (or it never sent the key). No collective
                    context. Its uploads go to the same field, which still
                    decays and reasons (so coverage is comparable).

Per upload (both methods, read-only, before the write) the state-gain class
of the candidate against the collective state at that moment
(field_sim.selection.decide_deposition): novel / correction / reinforce /
refresh / redundant. "redundant" = the collective state already held that
value with enough confidence and freshness - a pure repeat. Also recorded:
self-report (subject == sender) vs other entity; and same-round repeats (the
same key and object was already uploaded earlier in the same round).

Quality: coverage / precision of the field's unique triples against the
reference closure (field_sim.evaluation), last 20 rounds, seed means then
mean/SD over seeds. Costs: totals over 80 rounds per seed, mean/SD over seeds.

Outputs: field_sim/experiments/results_additional/agent_count/
Run: python -m field_sim.experiments.exp_agent_count
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
from field_sim.change_triggered import ChangeTriggeredPolicy
from field_sim.environment import generate_instances
from field_sim.evaluation import edge_triples, reference_state
from field_sim.experiments import exp_ccg_vs_fragmentation as ccg
from field_sim.experiments.harness import RESULTS_DIR, ExperimentConfig, build_world
from field_sim.reasoning import FieldReasoner
from field_sim.selection import decide_deposition

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "agent_count")
NS = [4, 8, 16, 32]
RATES = {0.02: "Slow", 0.10: "Moderate", 0.35: "Quick"}
SEEDS = list(range(8))
METHODS = ["proposed", "change_triggered"]
RADIUS = 26.0
CLASSES = ["novel", "correction", "reinforce", "refresh", "redundant"]


def instances(n: int) -> dict:
    return generate_instances(num_humanoids=n, num_pedestrians=8, num_areas=4, num_rooms=4, num_corridors=4,
                              num_workzones=2, num_passages=2)


assert instances(8) == ccg.INSTANCES, "N = 8 must be Case Study 1's population"


def cfg(n: int, rate: float, seed: int, steps: int = ccg.STEPS) -> ExperimentConfig:
    return ExperimentConfig(instances=instances(n), observation_radius=RADIUS, seed=seed,
                            evaporation_rate=ccg.EVAPORATION_RATE, change_prob=rate,
                            agent_move_step=ccg.AGENT_MOVE_STEP, steps=steps)


def run_one(n: int, rate: float, method: str, seed: int, steps: int = ccg.STEPS) -> dict:
    env, kb, agents, field, _ = build_world(cfg(n, rate, seed, steps), policy_factory=None)
    m1 = ccg._m1_policy_factory(kb) if method == "proposed" else None
    ct = ChangeTriggeredPolicy() if method == "change_triggered" else None
    reasoner = FieldReasoner()
    stream = hashlib.sha256()
    rows = []
    for t in range(1, steps + 1):
        env.step(t)
        sync_agent_locations(agents, env)
        cand = {}
        for a in agents:
            a.move(env.area_size)
            cand[a.agent_id] = a.form_candidate_traces(a.observe(env, kb, t), kb, t)
        stream.update(ccg.candidate_fingerprint(t, [c for a in agents for c in cand[a.agent_id]]).encode())
        cnt = Counter()
        sent_this_round = {}
        for a in agents:
            for c in cand[a.agent_id]:
                send = m1.evaluate(c, field, t, kb).transmit if m1 else ct.should_send(c)
                if not send:
                    continue
                cls = decide_deposition(c, field, t).classification
                who = "self" if c.subject == c.agent_id else "other"
                cnt["uploads"] += 1
                cnt[f"cls_{cls}"] += 1
                cnt[f"{who}_uploads"] += 1
                cnt[f"{who}_cls_{cls}"] += 1
                if sent_this_round.get(c.key) == c.object:
                    cnt["same_round_repeat"] += 1
                sent_this_round[c.key] = c.object
                field.set(c, t)
                if ct:
                    ct.record_sent(c)
        field.decay(t)
        field.reason(kb, t)
        truth = edge_triples(reference_state(env, kb, t, reasoner))
        held = edge_triples(field.edges)
        matched = len(held & truth)
        rows.append({"t": t, **cnt, "candidates": sum(len(v) for v in cand.values()),
                     "coverage": matched / len(truth), "precision": matched / len(held) if held else math.nan,
                     "truth_size": len(truth), "held": len(held), "matched": matched})
    return {"rounds": rows, "stream_sha": stream.hexdigest(), "n_agents": len(agents)}


# ------------------------------------------------------------------ driver
def _job(args):
    t0 = time.time()
    out = run_one(*args)
    out["seconds"] = time.time() - t0
    return args, out


def _ms(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (math.nan, math.nan)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(rows)


COUNT_KEYS = (["uploads", "candidates", "same_round_repeat", "self_uploads", "other_uploads"]
              + [f"cls_{c}" for c in CLASSES] + [f"{w}_cls_{c}" for w in ("self", "other") for c in CLASSES])


def seed_rows(runs):
    out = []
    for (n, rate, method, seed), o in sorted(runs.items()):
        R, last = o["rounds"], o["rounds"][-ccg.STEADY:]
        tot = {k: sum(r.get(k, 0) for r in R) for k in COUNT_KEYS}
        prec = [r["precision"] for r in last if not math.isnan(r["precision"])]
        truth = statistics.mean(r["truth_size"] for r in last)
        out.append({"N": n, "rate": RATES[rate], "change_prob": rate, "method": method, "seed": seed,
                    "agents": o["n_agents"], **tot,
                    "change_driven_uploads": tot["cls_novel"] + tot["cls_correction"],
                    "maintenance_uploads": tot["cls_reinforce"] + tot["cls_refresh"],
                    "uploads_per_agent": tot["uploads"] / o["n_agents"],
                    "uploads_round1": R[0].get("uploads", 0),
                    "truth_size_last20": truth,
                    "uploads_per_truth_triple": tot["uploads"] / truth,
                    "coverage_last20": statistics.mean(r["coverage"] for r in last),
                    "precision_last20": statistics.mean(prec) if prec else math.nan,
                    "held_last20": statistics.mean(r["held"] for r in last),
                    "seconds": round(o["seconds"], 1)})
    return out


def condition_rows(srows):
    by = defaultdict(list)
    for r in srows:
        by[(r["N"], r["change_prob"], r["method"])].append(r)
    out = []
    for (n, rate, method), rs in sorted(by.items()):
        row = {"N": n, "rate": RATES[rate], "change_prob": rate, "method": method, "seeds": len(rs)}
        for m in [k for k in rs[0] if k not in ("N", "rate", "change_prob", "method", "seed")]:
            row[f"{m}_mean"], row[f"{m}_sd"] = _ms([r[m] for r in rs])
        out.append(row)
    return out


def validate(runs, srows):
    checks = {"candidate_stream_identical_across_methods": all(
        runs[(n, r, "proposed", s)]["stream_sha"] == runs[(n, r, "change_triggered", s)]["stream_sha"]
        for (n, r, m, s) in runs if m == "proposed" and (n, r, "change_triggered", s) in runs),
        "agents_equal_humanoids": all(o["n_agents"] == n for (n, _, _, _), o in runs.items())}
    with open(os.path.join(RESULTS_DIR, "ccg_vs_fragmentation_summary.json"), encoding="utf-8") as f:
        saved = {r["seed"]: r for r in json.load(f)["per_run"] if r["policy"] == "selective" and r["radius"] == RADIUS}
    mine = {r["seed"]: r for r in srows if r["N"] == 8 and r["change_prob"] == ccg.CHANGE_PROB and r["method"] == "proposed"}
    if mine:
        checks["N8_p002_proposed_reproduces_case_study_1"] = all(
            mine[s]["uploads"] == saved[s]["cost"] for s in mine)
        checks["N8_p002_reproduction_detail"] = {s: (mine[s]["uploads"], saved[s]["cost"]) for s in sorted(mine)}
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    ap.add_argument("--ns", type=int, nargs="+", default=NS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"config{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"N": args.ns, "agents": "one per Humanoid (N Humanoids); 8 Pedestrians; fixed 4x4 region grid",
                   "methods": METHODS, "change_probs": args.rates, "radius": RADIUS, "seeds": SEEDS,
                   "steps": ccg.STEPS, "evaporation_rate": ccg.EVAPORATION_RATE, "min_confidence": 0.05,
                   "agent_move_step": ccg.AGENT_MOVE_STEP, "policy": "M1 threshold 0.5",
                   "change_triggered": "send iff object differs from this agent's last send for the key",
                   "upload_budget": None, "quality_window": ccg.STEADY}, f, indent=2)
    jobs = [(n, r, m, s) for n in args.ns for r in args.rates for m in METHODS for s in SEEDS]
    jobs.sort(key=lambda j: (-j[0], -j[1], j[2] != "proposed"))          # heaviest first
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        runs = dict(ex.map(_job, jobs))
    elapsed = time.time() - t0
    _write_csv(os.path.join(OUT_DIR, f"roundwise{args.tag}.csv"),
               [{"N": n, "rate": RATES[r], "method": m, "seed": s, **row}
                for (n, r, m, s), o in sorted(runs.items()) for row in o["rounds"]])
    srows = seed_rows(runs)
    _write_csv(os.path.join(OUT_DIR, f"seed_summary{args.tag}.csv"), srows)
    crows = condition_rows(srows)
    _write_csv(os.path.join(OUT_DIR, f"condition_summary{args.tag}.csv"), crows)
    checks = validate(runs, srows)
    with open(os.path.join(OUT_DIR, f"summary{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"validation": checks, "elapsed_seconds": elapsed}, f, indent=2)
    print(json.dumps(checks, indent=1), f"\nelapsed {elapsed / 60:.1f} min")
    for r in crows:
        print(f"{r['rate']:8s} N={r['N']:2d} {r['method']:16s} up {r['uploads_mean']:7.0f}±{r['uploads_sd']:5.0f}"
              f" chg {r['change_driven_uploads_mean']:6.0f} maint {r['maintenance_uploads_mean']:6.0f}"
              f" red {r['cls_redundant_mean']:6.0f} rep {r['same_round_repeat_mean']:5.0f}"
              f" self {r['self_uploads_mean']:6.0f} truth {r['truth_size_last20_mean']:5.0f}"
              f" cov {r['coverage_last20_mean']:.3f}±{r['coverage_last20_sd']:.3f} prec {r['precision_last20_mean']:.3f}")


if __name__ == "__main__":
    main()
