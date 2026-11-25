"""Inference provenance analysis for Case Study 1 (exp_ccg_vs_fragmentation):
what fraction of the inferred triples held by the collective state have at
least one recorded derivation path whose observation leaves come from two or
more different agents (field_sim.provenance for the exact criterion).

Re-runs the proposed policy ("selective": M1 + field reasoning) with the
exact Case Study 1 configuration (exp_ccg_vs_fragmentation._cfg/_m1_policy_
factory, seeds 0-7, 80 rounds) and a ProvenanceRecorder attached to the
field. Nothing about the simulation changes: transmission, rules, state
update, confidence, decay, reference state and coverage are the Case Study 1
code paths; the recorder is only notified of writes. validate() checks this
by requiring every run to reproduce the saved Case Study 1 per-run results
(results_main/ccg_vs_fragmentation_summary.json) exactly.

Measured at the same point as Case Study 1's evaluation (after transmit ->
decay -> reason, rounds 61-80), per (seed, radius, round), over the field's
currently stored edges with fact_type == "inferred" (one row per distinct
triple, never per firing).

This shows whether observations from several agents were actually combined
in the recorded derivations; it does NOT test whether any single agent could
have derived the same conclusion alone.

Outputs (field_sim/experiments/results_additional/provenance/):
  inference_provenance{tag}_roundwise.csv     seed x radius x round
  inference_provenance{tag}_by_property.csv   seed x radius x round x relation
  inference_provenance{tag}_by_radius.csv     summary (seed means -> mean/SD over seeds)
  inference_provenance{tag}_property_summary.csv
  inference_provenance{tag}_examples.json     r=26 multi-agent derivation trees
  inference_provenance{tag}_summary.json      criteria, validation, summaries

Run: python -m field_sim.experiments.exp_inference_provenance [--radii 26 --tag _r26_check]
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from field_sim.experiments import exp_ccg_vs_fragmentation as ccg
from field_sim.experiments.harness import RESULTS_DIR
from field_sim.provenance import MULTI, UNRESOLVED, ProvenanceRecorder

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "provenance")
SAVED_SUMMARY = os.path.join(RESULTS_DIR, "ccg_vs_fragmentation_summary.json")
POLICY = "selective"
ALL_RADII = sorted({r for _, r in ccg.FRAG + ccg.EXTRA_RADII})
EXAMPLE_RADIUS = 26.0
EXAMPLE_RELATIONS = ("shouldSlowDown", "potentialConflict", "pathBlocked", "shouldReroute")

COUNTS = ["n_inferred", "n_multi_agent_inferred", "n_correct_inferred", "n_correct_multi_agent_inferred",
          "n_unresolved", "n_multi_with_single_path", "n_inferred_latest_write_observed"]

# rule that derives each inferred relation, its antecedent count, and how many
# DISTINCT observed triples one path can bottom out in (structural, from the rules)
RULE_INFO = {
    "near": ("near_symmetric", 1, "1"),
    "approachingArea": ("approaching_area", 1, "1"),
    "canInteractWith": ("can_interact", 1, "1"),
    "pathBlocked": ("path_blocked", 2, "2"),
    "shouldReroute": ("should_reroute", 1, "2 (via pathBlocked)"),
    "potentialConflict": ("potential_conflict", 2, "2 (locatedAt + movingToward via approachingArea)"),
    "shouldSlowDown": ("should_slow_down", 2, "2-3 (potentialConflict's leaves + locatedAt)"),
}


def _frac(num, den):
    return num / den if den else math.nan


class ProvenanceObserver:
    """exp_ccg_vs_fragmentation.run_roi_with_inference observer: classifies
    the stored inferred edges at every scored round."""

    def __init__(self, seed, radius, keep_examples):
        self.seed, self.radius, self.keep_examples = seed, radius, keep_examples
        self.recorder = ProvenanceRecorder()
        self.rows, self.prop_rows, self.examples = [], [], []
        self.recorder_mismatches = 0

    def attach(self, field):
        field.provenance = self.recorder

    def on_round(self, t, field, truth):
        rec = self.recorder
        total, by_rel = Counter(), defaultdict(Counter)
        for key, edge in field.edges.items():
            # the recorder must have seen every write: its latest version of each stored key is the stored triple
            vid = rec.current.get(key)
            if vid is None or rec.versions[vid].triple != edge.triple:
                self.recorder_mismatches += 1
            if edge.fact_type != "inferred":
                continue
            c = rec.classify(edge)
            correct = edge.triple in truth
            inc = Counter(n_inferred=1)
            inc["n_multi_agent_inferred"] += c.status == MULTI
            inc["n_correct_inferred"] += correct
            inc["n_correct_multi_agent_inferred"] += correct and c.status == MULTI
            inc["n_unresolved"] += c.status == UNRESOLVED
            inc["n_multi_with_single_path"] += c.also_single
            inc["n_inferred_latest_write_observed"] += rec.versions[vid].kind == "observed" if vid is not None else 0
            total.update(inc)
            by_rel[edge.relation].update(inc)
            if (self.keep_examples and t == ccg.STEPS and c.status == MULTI
                    and edge.relation in EXAMPLE_RELATIONS):
                self.examples.append({"seed": self.seed, "round": t, "relation": edge.relation,
                                      "triple": list(edge.triple), "correct": correct,
                                      "also_has_single_agent_path": c.also_single,
                                      "derivation": rec.tree(c.vid)})
        base = {"seed": self.seed, "radius": self.radius, "round": t}
        self.rows.append({**base, **{k: total[k] for k in COUNTS},
                          "multi_agent_fraction": _frac(total["n_multi_agent_inferred"], total["n_inferred"]),
                          "correct_multi_agent_fraction": _frac(total["n_correct_multi_agent_inferred"],
                                                                total["n_correct_inferred"])})
        for rel in sorted(by_rel):
            cnt = by_rel[rel]
            self.prop_rows.append({**base, "relation": rel, **{k: cnt[k] for k in COUNTS},
                                   "multi_agent_fraction": _frac(cnt["n_multi_agent_inferred"], cnt["n_inferred"]),
                                   "correct_multi_agent_fraction": _frac(cnt["n_correct_multi_agent_inferred"],
                                                                         cnt["n_correct_inferred"])})


def _job(args):
    radius, seed = args
    obs = ProvenanceObserver(seed, radius, keep_examples=radius == EXAMPLE_RADIUS)
    factory, reasoning_enabled = ccg.POLICIES[POLICY]
    res = ccg.run_roi_with_inference(ccg._cfg(radius, seed), factory, reasoning_enabled, observer=obs)
    return radius, seed, res, obs.rows, obs.prop_rows, obs.examples, obs.recorder_mismatches


def summarize(rows, group_keys):
    """Per seed: mean over the window's rounds (NaN rounds skipped, valid rounds counted);
    then mean and SD over seeds. Counts: per-seed round means, then mean over seeds."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[tuple(r[k] for k in group_keys)][r["seed"]].append(r)
    out = []
    for gkey, by_seed in sorted(groups.items()):
        row = dict(zip(group_keys, gkey))
        for k in COUNTS:
            row[f"{k}_mean"] = statistics.mean(statistics.mean(x[k] for x in rs) for rs in by_seed.values())
        for f in ("multi_agent_fraction", "correct_multi_agent_fraction"):
            seed_means, valid = [], []
            for rs in by_seed.values():
                vals = [x[f] for x in rs if not math.isnan(x[f])]
                valid.append(len(vals))
                if vals:
                    seed_means.append(statistics.mean(vals))
            row[f"{f}_mean"] = statistics.mean(seed_means) if seed_means else math.nan
            row[f"{f}_sd"] = statistics.stdev(seed_means) if len(seed_means) > 1 else math.nan
            row[f"{f}_n_seeds"] = len(seed_means)
            row[f"{f}_valid_rounds_mean"] = statistics.mean(valid)
        row["is_lower_bound"] = row["n_unresolved_mean"] > 0
        row["n_seeds"] = len(by_seed)
        out.append(row)
    return out


def validate(runs, mismatches):
    """Every run must reproduce the saved Case Study 1 result exactly."""
    with open(SAVED_SUMMARY, encoding="utf-8") as f:
        saved = {(r["radius"], r["seed"]): r for r in json.load(f)["per_run"] if r["policy"] == POLICY}
    fields = ("cost", "ckc_field", "ckc_best", "ccg", "uploads", "window_mean", "coverage_by_relation",
              "field_replacements", "field_inference_conflicts")
    diffs = []
    for (radius, seed), res in runs.items():
        ref = saved.get((radius, seed))
        if ref is None:
            diffs.append(f"r={radius} seed={seed}: no saved run")
            continue
        diffs += [f"r={radius} seed={seed}: {k} {res[k]!r} != {ref[k]!r}" for k in fields if res[k] != ref[k]]
    return {"runs_compared": len(runs), "identical_to_saved_case_study_1": not diffs, "differences": diffs[:20],
            "recorder_mismatched_edges": mismatches}


def _write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main(radii, tag=""):
    jobs = [(r, s) for r in radii for s in ccg.SEEDS]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = list(pool.map(_job, jobs))
    runs, rows, prop_rows, examples, mismatches = {}, [], [], [], 0
    for radius, seed, res, rws, prws, exs, mm in results:
        runs[(radius, seed)] = res
        rows += rws
        prop_rows += prws
        examples += exs
        mismatches += mm

    checks = validate(runs, mismatches)
    by_radius = summarize(rows, ["radius"])
    by_prop = summarize(prop_rows, ["radius", "relation"])
    for r in by_prop:
        rule, n_ante, leaves = RULE_INFO.get(r["relation"], ("?", "?", "?"))
        r.update(rule=rule, antecedents=n_ante, observation_triples_per_path=leaves)

    os.makedirs(OUT_DIR, exist_ok=True)
    _write_csv(f"inference_provenance{tag}_roundwise.csv", rows)
    _write_csv(f"inference_provenance{tag}_by_property.csv", prop_rows)
    _write_csv(f"inference_provenance{tag}_by_radius.csv", by_radius)
    _write_csv(f"inference_provenance{tag}_property_summary.csv", by_prop)
    # a few examples, preferring different relations / seeds
    picked, seen = [], set()
    for ex in sorted(examples, key=lambda e: (EXAMPLE_RELATIONS.index(e["relation"]), e["seed"])):
        if ex["relation"] not in seen and ex["correct"]:
            picked.append(ex)
            seen.add(ex["relation"])
    with open(os.path.join(OUT_DIR, f"inference_provenance{tag}_examples.json"), "w", encoding="utf-8") as f:
        json.dump(picked[:3], f, indent=1)
    with open(os.path.join(OUT_DIR, f"inference_provenance{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "policy": POLICY, "radii": radii, "seeds": ccg.SEEDS, "steps": ccg.STEPS,
            "window_rounds": [ccg.STEPS - ccg.STEADY + 1, ccg.STEPS],
            "unit": "distinct stored triple with fact_type == 'inferred' (never firings or cumulative conclusions)",
            "criterion": "multi-agent-supported = at least one recorded derivation path of the stored value "
                         "whose observation leaves include two different observed triples reported by "
                         "different agents; paths are never merged; premise versions are the ones used at "
                         "inference time; untraceable paths -> unresolved",
            "aggregation": "per seed mean over rounds 61-80 (NaN rounds skipped), then mean/SD over seeds",
            "validation": checks, "by_radius": by_radius,
        }, f, indent=1, default=str)

    print(f"validation: {checks}")
    print(f"{'r':>4} {'n_inf':>7} {'n_multi':>8} {'frac':>13} {'n_corr':>7} {'n_cmulti':>8} {'corr frac':>13} "
          f"{'unres':>6} {'m+s':>6} {'valid':>6}")
    for r in by_radius:
        print(f"{r['radius']:>4.0f} {r['n_inferred_mean']:>7.1f} {r['n_multi_agent_inferred_mean']:>8.1f} "
              f"{r['multi_agent_fraction_mean']:>6.3f}±{r['multi_agent_fraction_sd']:.3f} "
              f"{r['n_correct_inferred_mean']:>7.1f} {r['n_correct_multi_agent_inferred_mean']:>8.1f} "
              f"{r['correct_multi_agent_fraction_mean']:>6.3f}±{r['correct_multi_agent_fraction_sd']:.3f} "
              f"{r['n_unresolved_mean']:>6.2f} {r['n_multi_with_single_path_mean']:>6.2f} "
              f"{r['multi_agent_fraction_valid_rounds_mean']:>6.1f}")
    if not checks["identical_to_saved_case_study_1"] or mismatches:
        raise RuntimeError("provenance run does not reproduce Case Study 1 - results not trustworthy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="*", default=ALL_RADII)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(sorted(a.radii), a.tag)
