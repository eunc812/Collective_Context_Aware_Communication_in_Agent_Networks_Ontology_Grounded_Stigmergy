"""Reasoning gain as an upload PRIORITY under a per-round budget: does
ranking by state gain + reasoning gain (COMBINED) detect potentialConflict
events earlier than ranking by state gain alone (STATE_ONLY)?

Controlled scheduling experiment (field_sim.budget_scheduler): every round,
all agents' candidates are pooled and at most B are uploaded by sequential
greedy selection, rescoring after each upload; ties use a seeded random
priority shared by both policies. Both policies then run the same decay and
live collective reasoning. The gain definitions, rules, single-pass live
reasoning, decay, confidence and cardinality are the existing ones.

World: the Case Study 1/2 configuration (exp_ccg_vs_fragmentation.INSTANCES,
r = 26, evaporation 0.15, agent_move_step 6.0, 80 rounds, seeds 0-7) at the
change probabilities of exp_vcr_current.ENV_RATES (0.02 / 0.10 / 0.35).

Budgets (fixed before any policy result was seen): the representative
per-round upload count of the existing UNRESTRICTED proposed method at
r = 26, change probability 0.02 - the median over rounds 2-80 and seeds 0-7
of `transmitted` in results_main/selective_vs_random_grid_roundwise.
csv, which is 76 - times 25/50/75/100 %: B = 19, 38, 57, 76. The same B
values are used at every change rate. Round 1 (initial build-up, 126-210
uploads unrestricted) is capped by B like every other round.

Events and detection: field_sim.conflict_detection (reference-closure
events; detection = the field holds the inferred conflict with a recorded
derivation path whose observation leaves are all still valid).

Outputs: field_sim/experiments/results_additional/budget_conflict/ (see main()); no figure.

Run: python -m field_sim.experiments.exp_budget_conflict_latency --rates 0.02 --tag _pilot
     python -m field_sim.experiments.exp_budget_conflict_latency            (all rates)
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
from field_sim.budget_scheduler import COMBINED, STATE_ONLY, select_round, tie_priority
from field_sim.conflict_detection import ConflictTracker
from field_sim.evaluation import edge_triples, reference_state
from field_sim.experiments import exp_ccg_vs_fragmentation as ccg
from field_sim.experiments.harness import ExperimentConfig, build_world
from field_sim.provenance import ProvenanceRecorder
from field_sim.reasoning import FieldReasoner

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "budget_conflict")
RADIUS = 26.0
STEPS = 80
SEEDS = list(range(8))
RATES = {0.02: "Slow", 0.10: "Moderate", 0.35: "Quick"}
REPRESENTATIVE_UPLOADS = 76          # see module docstring
BUDGETS = [round(REPRESENTATIVE_UPLOADS * f) for f in (0.25, 0.50, 0.75, 1.00)]   # 19, 38, 57, 76
POLICIES = [STATE_ONLY, COMBINED]
FINAL_WINDOW = ccg.STEADY


def _digest(items) -> str:
    return hashlib.sha256(repr(items).encode()).hexdigest()


def run_one(change_prob, budget, policy, seed, steps=STEPS):
    cfg = ExperimentConfig(instances=ccg.INSTANCES, observation_radius=RADIUS, seed=seed,
                           evaporation_rate=ccg.EVAPORATION_RATE, change_prob=change_prob,
                           agent_move_step=ccg.AGENT_MOVE_STEP, steps=steps)
    environment, kb, agents, field, _ = build_world(cfg, policy_factory=None)
    recorder = ProvenanceRecorder()
    field.provenance = recorder
    tracker, ref_reasoner, sel_reasoner = ConflictTracker(), FieldReasoner(), FieldReasoner()
    rounds, step_rows, agg = [], [], Counter()
    enabled_when_differ, pick_rel_differ, so_rel_differ = Counter(), Counter(), Counter()

    for t in range(1, steps + 1):
        environment.step(t)
        sync_agent_locations(agents, environment)
        cands = []
        for agent in agents:
            agent.move(environment.area_size)
            cands += agent.form_candidate_traces(agent.observe(environment, kb, t), kb, t)
        prios = [tie_priority(seed, t, c.agent_id, c.triple) for c in cands]
        rr = select_round(field, cands, t, kb, budget, policy, prios, sel_reasoner, record_steps=True)
        if len(rr.uploaded) > budget:
            raise AssertionError(f"budget exceeded at t={t}")
        field.decay(t)
        field.reason(kb, t)

        truth = edge_triples(reference_state(environment, kb, t, ref_reasoner))
        tracker.on_round(t, truth, {f.triple for f in environment.facts.values()}, field, recorder)
        held = edge_triples(field.edges)
        matched = len(held & truth)
        rounds.append({
            "round": t, "candidates": len(cands), "uploads": len(rr.uploaded),
            "coverage": matched / len(truth), "precision": matched / len(held) if held else math.nan,
            "stream": ccg.candidate_fingerprint(t, cands),
            "upload_set": _digest(sorted((cands[i].agent_id, cands[i].triple) for i in rr.uploaded)),
            "held": _digest(sorted(held)),
        })
        for s in rr.steps:
            agg["steps"] += 1
            agg["redundant_admitted"] += s.pick_state_gain <= 0.5
            agg["rg_positive"] += s.pick_reasoning_gain > 0
            if s.differs:
                agg["differ"] += 1
                pick_rel_differ[s.pick_relation] += 1
                so_rel_differ[s.state_only_relation or "(none eligible)"] += 1
                enabled_when_differ.update(s.enabled)
                step_rows.append({"round": t, "step": s.step, "pick_relation": s.pick_relation,
                                  "pick_classification": s.pick_classification,
                                  "pick_state_gain": s.pick_state_gain,
                                  "pick_reasoning_gain": s.pick_reasoning_gain,
                                  "enabled_relations": ";".join(f"{r}:{n}" for r, n in sorted(s.enabled.items())),
                                  "state_only_relation": s.state_only_relation or ""})

    events = tracker.finalize()
    window = rounds[-FINAL_WINDOW:]
    return {
        "change_prob": change_prob, "budget": budget, "policy": policy, "seed": seed,
        "events": events, "rounds": rounds, "step_rows": step_rows,
        "uploads": sum(r["uploads"] for r in rounds),
        "max_round_uploads": max(r["uploads"] for r in rounds),
        "coverage_final": statistics.mean(r["coverage"] for r in window),
        "precision_final": statistics.mean(r["precision"] for r in window),
        "divergence": {**agg, "pick_relation_when_differ": dict(pick_rel_differ),
                       "state_only_relation_when_differ": dict(so_rel_differ),
                       "enabled_when_differ": dict(enabled_when_differ)},
        "stale_leaf_violations": tracker.stale_leaf_violations,
    }


def _job(args):
    t0 = time.perf_counter()
    res = run_one(*args)
    res["seconds"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------- summaries
def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else math.nan


def _sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else math.nan


def _n(xs):
    return sum(1 for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))


SCOPES = {"new": lambda e: not e["initial"], "initial": lambda e: e["initial"], "all": lambda e: True}


def event_rows(runs):
    out = []
    for res in runs:
        for e in res["events"]:
            out.append({"rate": RATES[res["change_prob"]], "change_prob": res["change_prob"], "budget": res["budget"],
                        "policy": res["policy"], "seed": res["seed"], "subject": e["triple"][0],
                        "object": e["triple"][2], "t0": e["t0"], "t_end": e["t_end"], "duration": e["duration"],
                        "initial": e["initial"], "censored": e["censored"], "detected": e["detected"],
                        "td": e["td"], "latency": e["latency"], "missed": e["missed"],
                        "missed_censored": e["missed_censored"]})
    return out


def seed_rows(runs):
    out = []
    for res in runs:
        rounds = res["rounds"]
        base = {"rate": RATES[res["change_prob"]], "change_prob": res["change_prob"], "budget": res["budget"],
                "policy": res["policy"], "seed": res["seed"], "uploads": res["uploads"],
                "budget_utilization": res["uploads"] / (res["budget"] * len(rounds)),
                "budget_utilization_rounds2plus": sum(r["uploads"] for r in rounds[1:]) / (res["budget"] * (len(rounds) - 1)),
                "rounds_at_budget": sum(r["uploads"] == res["budget"] for r in rounds),
                "coverage_final": res["coverage_final"], "precision_final": res["precision_final"]}
        for scope, keep in SCOPES.items():
            ev = [e for e in res["events"] if keep(e)]
            det = [e for e in ev if e["detected"]]
            lat = [e["latency"] for e in det]
            out.append({**base, "scope": scope, "n_events": len(ev), "n_detected": len(det),
                        "n_missed": len(ev) - len(det),
                        "n_missed_censored": sum(e["missed_censored"] for e in ev),
                        "detection_rate": len(det) / len(ev) if ev else math.nan,
                        "latency_mean": statistics.mean(lat) if lat else math.nan,
                        "latency_median": statistics.median(lat) if lat else math.nan,
                        "duration_mean": statistics.mean(e["duration"] for e in ev) if ev else math.nan})
    return out


def paired(runs):
    """Same event under both policies (same rate, budget, seed)."""
    by = {(r["change_prob"], r["budget"], r["seed"], r["policy"]): r for r in runs}
    ev_rows, seed_out = [], []
    for (cp, b, s, pol), res in sorted(by.items()):
        if pol != STATE_ONLY or (cp, b, s, COMBINED) not in by:
            continue
        so = {(tuple(e["triple"]), e["t0"]): e for e in res["events"]}
        co = {(tuple(e["triple"]), e["t0"]): e for e in by[(cp, b, s, COMBINED)]["events"]}
        if set(so) != set(co):
            raise AssertionError(f"event lists differ between policies at p={cp}, B={b}, seed={s}")
        for key in sorted(so):
            a, c = so[key], co[key]
            status = ("both" if a["detected"] and c["detected"] else "only_state_only" if a["detected"]
                      else "only_combined" if c["detected"] else "neither")
            ev_rows.append({"rate": RATES[cp], "change_prob": cp, "budget": b, "seed": s, "subject": key[0][0],
                            "object": key[0][2], "t0": key[1], "duration": a["duration"], "initial": a["initial"],
                            "censored": a["censored"], "state_only_latency": a["latency"],
                            "combined_latency": c["latency"], "status": status,
                            "latency_diff_combined_minus_state_only":
                                c["latency"] - a["latency"] if status == "both" else None})
        for scope, keep in SCOPES.items():
            rows = [r for r in ev_rows if r["change_prob"] == cp and r["budget"] == b and r["seed"] == s
                    and keep(r)]
            diffs = [r["latency_diff_combined_minus_state_only"] for r in rows if r["status"] == "both"]
            st = Counter(r["status"] for r in rows)
            seed_out.append({"rate": RATES[cp], "change_prob": cp, "budget": b, "seed": s, "scope": scope,
                             "n_events": len(rows), "n_both": st["both"], "n_only_state_only": st["only_state_only"],
                             "n_only_combined": st["only_combined"], "n_neither": st["neither"],
                             "latency_diff_mean": statistics.mean(diffs) if diffs else math.nan,
                             "latency_diff_median": statistics.median(diffs) if diffs else math.nan})
    return ev_rows, seed_out


def condition_summary(seed_rws, paired_seed):
    out = []
    groups = defaultdict(list)
    for r in seed_rws:
        groups[(r["change_prob"], r["budget"], r["policy"], r["scope"])].append(r)
    for (cp, b, pol, scope), rs in sorted(groups.items()):
        row = {"rate": RATES[cp], "change_prob": cp, "budget": b, "policy": pol, "scope": scope, "n_seeds": len(rs)}
        for k in ("n_events", "n_detected", "n_missed", "n_missed_censored", "detection_rate", "latency_mean",
                  "latency_median", "duration_mean", "uploads", "budget_utilization",
                  "budget_utilization_rounds2plus", "rounds_at_budget", "coverage_final", "precision_final"):
            vals = [r[k] for r in rs]
            row[f"{k}_mean"] = _mean(vals)
            row[f"{k}_sd"] = _sd(vals)
            row[f"{k}_n_seeds"] = _n(vals)
        row["n_events_total"] = sum(r["n_events"] for r in rs)
        out.append(row)
    pout = []
    pg = defaultdict(list)
    for r in paired_seed:
        pg[(r["change_prob"], r["budget"], r["scope"])].append(r)
    for (cp, b, scope), rs in sorted(pg.items()):
        d = [r["latency_diff_mean"] for r in rs]
        pout.append({"rate": RATES[cp], "change_prob": cp, "budget": b, "scope": scope,
                     "events_total": sum(r["n_events"] for r in rs),
                     "both_total": sum(r["n_both"] for r in rs),
                     "only_state_only_total": sum(r["n_only_state_only"] for r in rs),
                     "only_combined_total": sum(r["n_only_combined"] for r in rs),
                     "neither_total": sum(r["n_neither"] for r in rs),
                     "seed_latency_diff_mean": _mean(d), "seed_latency_diff_sd": _sd(d),
                     "seeds_with_paired_events": _n(d),
                     "seeds_diff_negative": sum(1 for x in d if not math.isnan(x) and x < 0),
                     "seeds_diff_positive": sum(1 for x in d if not math.isnan(x) and x > 0),
                     "seeds_diff_zero": sum(1 for x in d if not math.isnan(x) and x == 0)})
    return out, pout


def divergence_rows(runs):
    """Same-state ranking difference (combined runs) and trajectory divergence (state-only vs combined)."""
    by = {(r["change_prob"], r["budget"], r["seed"], r["policy"]): r for r in runs}
    out = []
    for (cp, b, s, pol), res in sorted(by.items()):
        if pol != COMBINED:
            continue
        d = res["divergence"]
        row = {"rate": RATES[cp], "change_prob": cp, "budget": b, "seed": s, "steps": d.get("steps", 0),
               "steps_differ": d.get("differ", 0),
               "fraction_differ": d.get("differ", 0) / d["steps"] if d.get("steps") else math.nan,
               "picks_with_reasoning_gain": d.get("rg_positive", 0),
               "picks_state_gain_below_threshold": d.get("redundant_admitted", 0),
               "pick_relation_when_differ": json.dumps(d["pick_relation_when_differ"], sort_keys=True),
               "state_only_relation_when_differ": json.dumps(d["state_only_relation_when_differ"], sort_keys=True),
               "enabled_when_differ": json.dumps(d["enabled_when_differ"], sort_keys=True)}
        other = by.get((cp, b, s, STATE_ONLY))
        if other:
            ups = [x["upload_set"] != y["upload_set"] for x, y in zip(res["rounds"], other["rounds"])]
            held = [x["held"] != y["held"] for x, y in zip(res["rounds"], other["rounds"])]
            row.update(rounds_upload_set_differs=sum(ups),
                       first_round_upload_set_differs=ups.index(True) + 1 if any(ups) else None,
                       rounds_state_differs=sum(held),
                       first_round_state_differs=held.index(True) + 1 if any(held) else None)
        out.append(row)
    return out


def validate(runs):
    streams, events = defaultdict(set), defaultdict(set)
    for r in runs:
        streams[(r["change_prob"], r["seed"])].add(tuple(x["stream"] for x in r["rounds"]))
        events[(r["change_prob"], r["seed"])].add(
            tuple((tuple(e["triple"]), e["t0"], e["t_end"]) for e in r["events"]))
    return {
        "budget_never_exceeded": all(r["max_round_uploads"] <= r["budget"] for r in runs),
        "candidate_stream_identical_across_policies_and_budgets": all(len(v) == 1 for v in streams.values()),
        "event_list_identical_across_policies_and_budgets": all(len(v) == 1 for v in events.values()),
        "stale_leaf_violations": sum(r["stale_leaf_violations"] for r in runs),
        "runs": len(runs),
    }


def _write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(rates, tag=""):
    jobs = [(cp, b, pol, s) for cp in rates for b in BUDGETS for pol in POLICIES for s in SEEDS]
    jobs.sort(key=lambda j: (j[2] != COMBINED, -j[1]))          # longest (combined, large B) first
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        runs = list(pool.map(_job, jobs))
    checks = validate(runs)

    os.makedirs(OUT_DIR, exist_ok=True)
    seed_rws = seed_rows(runs)
    ev_paired, paired_seed = paired(runs)
    cond, cond_paired = condition_summary(seed_rws, paired_seed)
    _write_csv(f"events{tag}.csv", event_rows(runs))
    _write_csv(f"events_paired{tag}.csv", ev_paired)
    _write_csv(f"seed_summary{tag}.csv", seed_rws)
    _write_csv(f"paired_seed_summary{tag}.csv", paired_seed)
    _write_csv(f"condition_summary{tag}.csv", cond)
    _write_csv(f"paired_condition_summary{tag}.csv", cond_paired)
    _write_csv(f"selection_divergence{tag}.csv", divergence_rows(runs))
    _write_csv(f"selection_differences_log{tag}.csv",
               [{"rate": RATES[r["change_prob"]], "budget": r["budget"], "seed": r["seed"], **s}
                for r in runs if r["policy"] == COMBINED for s in r["step_rows"]])
    _write_csv(f"roundwise{tag}.csv",
               [{"rate": RATES[r["change_prob"]], "budget": r["budget"], "policy": r["policy"], "seed": r["seed"],
                 **{k: v for k, v in x.items() if k not in ("stream", "upload_set", "held")}}
                for r in runs for x in r["rounds"]])
    with open(os.path.join(OUT_DIR, f"summary{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "radius": RADIUS, "steps": STEPS, "seeds": SEEDS, "change_probs": rates,
            "representative_uploads": REPRESENTATIVE_UPLOADS, "budgets": BUDGETS,
            "budget_rule": "median per-round uploads (rounds 2-80, seeds 0-7) of the unrestricted proposed method "
                           "at r=26, p=0.02 (selective_vs_random_grid_roundwise.csv) x 25/50/75/100 %",
            "threshold": 0.5, "final_window": FINAL_WINDOW, "validation": checks,
            "runtime_seconds": {"total": time.perf_counter() - t_start,
                                "per_run_max": max(r["seconds"] for r in runs)},
        }, f, indent=1)
    print("validation:", checks)
    for r in cond:
        if r["scope"] == "new":
            print(f"{r['rate']:<8} B={r['budget']:>3} {r['policy']:<10} events {r['n_events_mean']:.2f} "
                  f"det {r['detection_rate_mean']:.3f}±{r['detection_rate_sd']:.3f} "
                  f"lat {r['latency_mean_mean']:.2f}±{r['latency_mean_sd']:.2f} (seeds {r['latency_mean_n_seeds']}) "
                  f"up {r['uploads_mean']:.0f} util {r['budget_utilization_mean']:.3f} "
                  f"cov {r['coverage_final_mean']:.3f} prec {r['precision_final_mean']:.3f}")
    for r in cond_paired:
        print(f"paired {r['rate']:<8} B={r['budget']:>3} {r['scope']:<7} events {r['events_total']} both "
              f"{r['both_total']} only_SO {r['only_state_only_total']} only_C {r['only_combined_total']} "
              f"neither {r['neither_total']} diff {r['seed_latency_diff_mean']:+.3f}±{r['seed_latency_diff_sd']:.3f} "
              f"(-/0/+ {r['seeds_diff_negative']}/{r['seeds_diff_zero']}/{r['seeds_diff_positive']})")
    critical = [k for k, v in checks.items() if v is False] + (["stale_leaf_violations"] if checks["stale_leaf_violations"] else [])
    if critical:
        raise RuntimeError(f"validation failed: {critical}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="*", default=sorted(RATES))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(sorted(a.rates), a.tag)
