"""Communication cost of state access versus the synchronization period K,
counted in messages (not bytes).

Model (Case Study 1 world: 8 Humanoid agents, r = 26, evaporation 0.15,
floor 0.05, 80 rounds, seeds 0-7, M1 threshold 0.5; p = 0.02 / 0.10 / 0.35):

  edge     holds the collective state E (field_sim.field.Field). Every upload
           is written to E immediately; E decays and runs one reasoning pass
           at the end of every round, exactly as in Case Study 1.
  agents   do NOT read E. Each agent a decides (M1Policy) against its own
           local copy L_a. Between synchronizations L_a changes only by
           (i) the agent's own uploads (written to L_a as well as to E) and
           (ii) the same decay and floor as E (Field.decay, every round). The
           local copies do not reason; inferred triples arrive by sync.
  sync     at the end of round t when t % K == 0 (after E's decay and
           reasoning), the edge broadcasts ONE delta that every agent
           receives, after which every L_a equals E. The delta is taken
           against P = the state of the last broadcast, decayed with the same
           rule (what a receiver can compute by itself): one entry per key
           that was added (in E, not in P), changed (in both, any field
           differs - i.e. the edge was written since the last sync) or
           deleted (in P, not in E). Pure decay is never sent, because P
           decays exactly like E.

Costs (messages, totals over the 80 rounds, mean/SD over seeds):
  uplink    = number of uploaded traces (all agents)
  downlink  = number of delta entries broadcast (counted once per broadcast,
              not per receiver); also reported: the number of broadcasts
  total     = uplink + downlink
K = 0 is free, instantaneous access (agents read E directly, no downlink):
Case Study 1 itself, validated per seed. No sync is forced at the last round
(80 % K rounds may end without one); downlink counts the broadcasts made.

Also per run: uplink classes against E just before the write
(field_sim.selection.decide_deposition: novel / correction / reinforce /
refresh / redundant), the delta's ADD / CHANGE / DELETE split, and the
edge's coverage / precision over the last 20 rounds (quality context; no K
is selected as "optimal").

Outputs: field_sim/experiments/results_additional/sync_messages/
Run: python -m field_sim.experiments.exp_sync_messages
"""

import argparse
import copy
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
from field_sim.experiments.harness import RESULTS_DIR, build_world
from field_sim.field import Field
from field_sim.reasoning import FieldReasoner
from field_sim.selection import decide_deposition

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "sync_messages")
KS = list(range(0, 21))                 # 0 = free access (Case Study 1)
RATES = {0.02: "Slow", 0.10: "Moderate", 0.35: "Quick"}
SEEDS = list(range(8))
RADIUS = 26.0
CLASSES = ["novel", "correction", "reinforce", "refresh", "redundant"]


def cfg(rate, seed, steps=ccg.STEPS):
    c = ccg._cfg(RADIUS, seed)
    c.change_prob = rate
    c.steps = steps
    return c


def snapshot(field: Field) -> Field:
    """An independent copy of a field's state with the same decay/floor rule."""
    f = Field(evaporation_rate=field.evaporation_rate, min_confidence=field.min_confidence)
    f.edges = copy.deepcopy(field.edges)
    return f


def delta(prev: dict, cur: dict) -> dict:
    """Entries needed to turn `prev` (the receivers' decayed last broadcast) into `cur`."""
    add = [k for k in cur if k not in prev]
    change = [k for k in cur if k in prev and cur[k] != prev[k]]
    delete = [k for k in prev if k not in cur]
    return {"add": add, "change": change, "delete": delete}


def apply_delta(prev: dict, cur: dict, d: dict) -> dict:
    """Receiver side: rebuild the state from `prev` and the delta's entries (values taken from `cur`)."""
    out = {k: v for k, v in prev.items() if k not in set(d["delete"])}
    for k in d["add"] + d["change"]:
        out[k] = cur[k]
    return out


def run_one(rate, k, seed, steps=ccg.STEPS, check=False):
    env, kb, agents, edge, _ = build_world(cfg(rate, seed, steps), policy_factory=None)
    m1 = ccg._m1_policy_factory(kb)
    local = {a.agent_id: snapshot(edge) for a in agents} if k else None
    prev = snapshot(edge) if k else None           # receivers' view of the last broadcast
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
        for a in agents:
            view = local[a.agent_id] if k else edge
            for c in cand[a.agent_id]:
                if not m1.evaluate(c, view, t, kb).transmit:
                    continue
                cnt["uplink"] += 1
                cnt[f"cls_{decide_deposition(c, edge, t).classification}"] += 1
                edge.set(c, t)
                if k:
                    local[a.agent_id].set(c, t)
        edge.decay(t)
        edge.reason(kb, t)
        if k:
            for f in local.values():
                f.decay(t)
            prev.decay(t)
            if t % k == 0:
                d = delta(prev.edges, edge.edges)
                if check:
                    assert apply_delta(prev.edges, edge.edges, d) == edge.edges
                n = sum(map(len, d.values()))
                cnt["syncs"] += 1
                cnt["downlink"] += n
                for typ in d:
                    cnt[f"dl_{typ}"] += len(d[typ])
                prev = snapshot(edge)
                local = {a.agent_id: snapshot(edge) for a in agents}
        truth = edge_triples(reference_state(env, kb, t, reasoner))
        held = edge_triples(edge.edges)
        matched = len(held & truth)
        rows.append({"t": t, **cnt, "coverage": matched / len(truth),
                     "precision": matched / len(held) if held else math.nan})
    return {"rounds": rows, "stream_sha": stream.hexdigest()}


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


COUNT_KEYS = ["uplink", "downlink", "syncs", "dl_add", "dl_change", "dl_delete"] + [f"cls_{c}" for c in CLASSES]


def seed_rows(runs):
    out = []
    for (rate, k, seed), o in sorted(runs.items()):
        R, last = o["rounds"], o["rounds"][-ccg.STEADY:]
        tot = {key: sum(r.get(key, 0) for r in R) for key in COUNT_KEYS}
        prec = [r["precision"] for r in last if not math.isnan(r["precision"])]
        out.append({"rate": RATES[rate], "change_prob": rate, "K": k, "seed": seed, **tot,
                    "total": tot["uplink"] + tot["downlink"],
                    "coverage_last20": statistics.mean(r["coverage"] for r in last),
                    "precision_last20": statistics.mean(prec) if prec else math.nan,
                    "seconds": round(o["seconds"], 1)})
    return out


def condition_rows(srows):
    by = defaultdict(list)
    for r in srows:
        by[(r["change_prob"], r["K"])].append(r)
    out = []
    for (rate, k), rs in sorted(by.items()):
        row = {"rate": RATES[rate], "change_prob": rate, "K": k, "seeds": len(rs)}
        for m in [x for x in rs[0] if x not in ("rate", "change_prob", "K", "seed")]:
            row[f"{m}_mean"], row[f"{m}_sd"] = _ms([r[m] for r in rs])
        out.append(row)
    return out


def validate(runs, srows):
    checks = {"candidate_stream_identical_across_K": all(
        len({o["stream_sha"] for (r, _, s), o in runs.items() if (r, s) == key}) == 1
        for key in {(r, s) for (r, _, s) in runs})}
    with open(os.path.join(RESULTS_DIR, "ccg_vs_fragmentation_summary.json"), encoding="utf-8") as f:
        saved = {r["seed"]: r["cost"] for r in json.load(f)["per_run"]
                 if r["policy"] == "selective" and r["radius"] == RADIUS}
    mine = {r["seed"]: r["uplink"] for r in srows if r["K"] == 0 and r["change_prob"] == ccg.CHANGE_PROB}
    if mine:
        checks["K0_p002_reproduces_case_study_1"] = all(mine[s] == saved[s] for s in mine)
    checks["K0_has_no_downlink"] = all(r["downlink"] == 0 for r in srows if r["K"] == 0)
    checks["syncs_equal_floor_steps_over_K"] = all(r["syncs"] == ccg.STEPS // r["K"] for r in srows if r["K"])
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"K": args.ks, "K0": "free instantaneous access (Case Study 1)", "change_probs": args.rates,
                   "agents": 8, "radius": RADIUS, "seeds": SEEDS, "steps": ccg.STEPS,
                   "evaporation_rate": ccg.EVAPORATION_RATE, "min_confidence": 0.05, "policy": "M1 threshold 0.5",
                   "local_copy": "own uploads + same decay/floor as the edge; replaced by the edge state at each sync",
                   "uplink_unit": "uploaded trace", "downlink_unit": "delta entry (ADD/CHANGE/DELETE), once per broadcast",
                   "quality_window": ccg.STEADY}, f, indent=2)
    jobs = [(r, k, s) for r in args.rates for k in args.ks for s in SEEDS]
    jobs.sort(key=lambda j: -j[0])                                    # quick (slowest) first
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        runs = dict(ex.map(_job, jobs))
    elapsed = time.time() - t0
    _write_csv(os.path.join(OUT_DIR, "roundwise.csv"),
               [{"rate": RATES[r], "K": k, "seed": s, **row} for (r, k, s), o in sorted(runs.items())
                for row in o["rounds"]])
    srows = seed_rows(runs)
    _write_csv(os.path.join(OUT_DIR, "seed_summary.csv"), srows)
    crows = condition_rows(srows)
    _write_csv(os.path.join(OUT_DIR, "condition_summary.csv"), crows)
    checks = validate(runs, srows)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"validation": checks, "elapsed_seconds": elapsed}, f, indent=2)
    print(json.dumps(checks, indent=1), f"\nelapsed {elapsed / 60:.1f} min")
    for r in crows:
        print(f"{r['rate']:8s} K={r['K']:2d} up {r['uplink_mean']:6.0f} down {r['downlink_mean']:6.0f} "
              f"total {r['total_mean']:6.0f}±{r['total_sd']:4.0f} syncs {r['syncs_mean']:3.0f} "
              f"red {r['cls_redundant_mean']:5.0f} cov {r['coverage_last20_mean']:.3f} "
              f"prec {r['precision_last20_mean']:.3f}")


if __name__ == "__main__":
    main()
