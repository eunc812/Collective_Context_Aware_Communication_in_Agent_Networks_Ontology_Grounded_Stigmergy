"""Re-run of the fig_roi.png experiment (Collective Cognition ROI) on the
CURRENT codebase, but over a larger, procedurally-generated VLA instance
population (field_sim.environment.generate_instances(), not the original
11-instance DEFAULT_INSTANCES), with fragmentation (observation_radius)
made visually explicit per point (a sequential color ramp) rather than
only a few text-annotated points on one aggregate curve.

  y = CCG = CKC_field - CKC_best_individual
      (the fraction of ground truth the POOLED field gets right, minus the
      fraction the single BEST individual agent gets right on its own -
      exactly "collected-in-one-place cognition ratio minus each
      individual's own cognition ratio")
  x = communication volume = total deposits sent this run (log scale)
      comparing M1 (selective deposition) against broadcast-all at the
      SAME fragmentation level - thin connecting segments make that
      per-radius cost gap visible directly, not just two separate curves.

Does NOT reuse exp_roi.run_roi() unchanged - see run_roi_with_inference()
below: the accuracy comparison here also covers the 6 inferred-only
relations, which run_roi()'s own comparison can never reflect at all
(field_sim.environment.Environment never generates ground truth for an
inferred=True property, so there's nothing to score against). Does NOT
touch figures/fig_roi.png or results/roi_*.csv/json (those stay exactly
as originally generated) - writes to new, separate files.

Coverage is triple-based (field_sim.evaluation): the reference state is the
environment's current observable facts closed under the reasoning rules,
multi-valued relations keep every object (field_sim.cardinality), and
coverage = |evaluated triples & reference triples| / |reference triples|.

Outputs: results_main/ccg_vs_fragmentation.csv + _summary.json (the
figure is drawn from the saved CSV by figures/make_ccg_vs_fragmentation.py)
Run: python -m field_sim.experiments.exp_ccg_vs_fragmentation
"""

import hashlib
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from field_sim.agent import sync_agent_locations
from field_sim.communication import M1Config, M1Policy, StateGainConfig, StateGainPolicy
from field_sim.environment import generate_instances
from field_sim.evaluation import edge_triples, matched_by_relation, memory_triples, reference_state, score
from field_sim.experiments.harness import ExperimentConfig, avg, build_world, write_csv, write_json
from field_sim.reasoning import FieldReasoner

# Copied from field_sim.experiments.exp_roi (not imported - that module's
# own run_roi() is no longer used here, only its 4 constants were, and
# exp_roi.py is the frozen fig_roi.png/roi_*.csv/json reference experiment,
# never touched by this file).
RA_THRESHOLD = 0.5     # M1's ReasoningAwarePolicy transmission threshold
FRAG = [("low", 60.0), ("", 45.0), ("medium", 34.0), ("", 26.0), ("", 20.0), ("high", 15.0)]
SEEDS = list(range(8))
STEADY = 20            # last N rounds of a run averaged into ckc_field/ckc_best

# 8 Humanoids (sensors) + 8 Pedestrians (observed-only) + 16 SpatialRegions
# (4 Area + 4 Room + 4 Corridor + 2 WorkZone + 2 Passage - a perfect square,
# so Environment's _partition_regions() tiles it as a gapless 4x4 grid with
# no padding needed) + the usual fixed background object counts. Kept in
# sync with exp_vcr_current.INSTANCES - same call, same population, in both
# files.
INSTANCES = generate_instances(
    num_humanoids=8, num_pedestrians=8,
    num_areas=4, num_rooms=4, num_corridors=4, num_workzones=2, num_passages=2,
)

# Field decay rate + environment change rate - tuned to the VCR-optimal
# decay found by field_sim.experiments.exp_vcr_current.best_decay() for its
# "slow" environment, instead of ExperimentConfig's untuned defaults
# (evaporation_rate=0.08, change_prob=0.03): best_decay(data, "slow") ==
# 0.15 keeps the field closest to the ideal (VCR=1, SCR=0) corner for this
# change rate. Re-tuned after field_sim.selection.decide_deposition()'s
# self-report bypass fix shifted the "slow" optimum down from 0.2 (its
# pre-fix value) to 0.15 - the bypass makes the field react faster to a
# self-reporting subject regardless of margin/staleness, so a lower decay
# no longer costs as much SCR as it used to for the same VCR.
# NOTE (2026-09-10): exp_vcr_current's "slow" environment itself just
# changed meaning again (change_prob now also gates drift occurrence, see
# field_sim/environment.py's step() and exp_vcr_current's module
# docstring) - this 0.15 was tuned against the PRIOR "slow" definition and
# has not yet been re-verified against the new one; re-run
# exp_vcr_current's sweep and re-check best_decay(data, "slow") before
# trusting this value again.
EVAPORATION_RATE = 0.15
# Matches exp_vcr_current.ENV_RATES' "slow" entry - the single knob
# "environment speed" now means end-to-end (change_prob gates drift
# occurrence too, not just carrying/interactingWith/non-drifting
# relations - see exp_vcr_current's module docstring for the full
# rationale).
CHANGE_PROB = 0.02

# Matches exp_vcr_current.FIXED_AGENT_MOVE_STEP - a single fixed magnitude
# (how far a drift step goes WHEN one happens) shared by every environment
# tier there now, no longer a per-tier "speed" value itself (change_prob
# alone plays that role now, via the CHANGE_PROB constant above).
AGENT_MOVE_STEP = 6.0

# Matches exp_vcr_current.STEPS - same run length on both sides of this
# pair of experiments.
STEPS = 80

# Below 15, ckc_field itself starts collapsing (not just ckc_best) - the
# collective layer runs out of raw material to stay accurate, and CCG turns
# back down. Included so the figure shows the full rise-then-fall shape,
# not just the rising half FRAG (exp_roi.py's own sweep) stops at.
EXTRA_RADII = [(f"below-15 (r={r:g})", r) for r in (12, 10, 8, 6)]


def candidate_fingerprint(t: int, candidates) -> str:
    """Hash of one round's candidate stream (every grounded trace of every
    agent, in agent order). Identical across policies for the same seed iff
    the environment, movement and sensing are unaffected by the policy."""
    return hashlib.sha256(repr([(t, c.subject, c.relation, c.object, c.confidence, c.agent_id)
                                for c in candidates]).encode()).hexdigest()


def run_roi_with_inference(cfg: ExperimentConfig, policy_factory=None, reasoning_enabled: bool = True,
                           observer=None) -> dict:
    """One run: transmit -> decay -> (reason) every round, scored over the
    last STEADY rounds with triple-based coverage (field_sim.evaluation).

      ckc_field = coverage of the field's triples against the reference state
      ckc_best  = per round, the highest coverage among the individuals' own
                  memories, averaged over the window
      ccg       = ckc_field - ckc_best

    Reference state: the environment's current observable facts plus their
    reasoning closure, rebuilt every round (field_sim.evaluation.
    reference_state) - it covers the inferred relations through the
    closure, so the field is credited for every currently-true inferred
    triple it holds and an individual (direct observation only) can never
    match one.

    Individuals: own[agent] maps (subject, relation) -> the latest object
    that agent observed directly - the pre-existing definition (direct
    observation only, never passed through any reasoner, no decay), kept
    deliberately pair-keyed: a sensor reports one value per observed fact
    (near = that subject's nearest entity), and the individual keeps the
    latest one.

    `policy_factory` (KnowledgeBase -> CommunicationPolicy, or None): same
    convention as field_sim.experiments.harness.build_world() - None means
    broadcast (every candidate transmitted unconditionally).

    `reasoning_enabled` (default True): collective reasoning ABLATION - when
    False, field.reason() is never called, so the field never holds an
    inferred triple. The transmission policy is untouched (M1's reasoning
    gain still evaluates candidates through its own counterfactual
    closure).

    `observer` (default None): analysis-only hook - observer.attach(field)
    right after the world is built, observer.on_round(t, field, truth) at
    every scored round, after the round's evaluation. Used by
    exp_inference_provenance; it must not write to the field."""
    environment, kb, agents, field, sim = build_world(cfg, policy_factory=policy_factory)
    if observer is not None:
        observer.attach(field)

    own = {a.agent_id: {} for a in agents}
    field_cov, best_ind = [], []
    deposits = 0
    reasoner = FieldReasoner()
    uploads = Counter()          # (relation, classification) of every transmitted candidate
    window = Counter()           # summed matched/missing/incorrect/truth_size over the window
    by_relation = Counter()      # (relation, "matched"|"truth") summed over the window
    stream = hashlib.sha256()

    for t in range(1, cfg.steps + 1):
        environment.step(t)
        sync_agent_locations(agents, environment)
        cand = {}
        for agent in agents:
            agent.move(environment.area_size)
            raws = agent.observe(environment, kb, t)
            cs = agent.form_candidate_traces(raws, kb, t)
            cand[agent.agent_id] = cs
            for c in cs:
                own[agent.agent_id][c.pair] = c.object
        stream.update(candidate_fingerprint(t, [c for a in agents for c in cand[a.agent_id]]).encode())
        for agent in agents:
            if sim.policy is not None:
                decisions = agent.communicate(field, cand[agent.agent_id], t, sim.policy, kb)
                for c, d in zip(cand[agent.agent_id], decisions):
                    if d.transmit:
                        deposits += 1
                        uploads[(c.relation, d.classification or "reasoning_only")] += 1
            else:
                for c in cand[agent.agent_id]:
                    field.set(c, t)
                    deposits += 1
                    uploads[(c.relation, "broadcast")] += 1
        field.decay(t)
        if reasoning_enabled:
            field.reason(kb, t)

        if t <= cfg.steps - STEADY:
            continue  # only score the steady-state window

        truth = edge_triples(reference_state(environment, kb, t, reasoner))
        if not truth:
            raise ValueError(f"empty reference state at t={t} - coverage undefined")
        held = edge_triples(field.edges)
        s = score(held, truth)
        field_cov.append(s["coverage"])
        best_ind.append(max(score(memory_triples(own[a.agent_id]), truth)["coverage"] for a in agents))
        window.update({k: s[k] for k in ("matched", "missing", "incorrect", "truth_size")})
        for rel, (m, n) in matched_by_relation(held, truth).items():
            by_relation[(rel, "matched")] += m
            by_relation[(rel, "truth")] += n
        if observer is not None:
            observer.on_round(t, field, truth)

    ckc_field = avg(field_cov)
    ckc_best = avg(best_ind)
    return {
        "ccg": ckc_field - ckc_best, "cost": deposits, "ckc_field": ckc_field, "ckc_best": ckc_best,
        "window_mean": {k: v / STEADY for k, v in window.items()},
        "coverage_by_relation": {
            rel: by_relation[(rel, "matched")] / by_relation[(rel, "truth")]
            for rel in sorted({r for r, _ in by_relation})
        },
        "truth_by_relation": {rel: by_relation[(rel, "truth")] / STEADY for rel in sorted({r for r, _ in by_relation})},
        "uploads": {f"{rel}|{cls}": n for (rel, cls), n in sorted(uploads.items())},
        "field_replacements": dict(field.replacements),
        "field_inference_conflicts": len(field.inference_conflicts),
        "reference_conflicts": reasoner.stats["single_valued_conflicts"],
        "reference_conflicts_by_relation": dict(reasoner.conflicts_by_relation),
        "reference_iteration_limit_hits": reasoner.stats["iteration_limit_hits"],
        "candidate_stream_sha256": stream.hexdigest(),
    }


def _m1_policy_factory(kb_):
    """The "selective" policy run_sweep() below has always used - state_gain
    + reasoning_gain combined (field_sim.communication.M1Policy), at this
    module's own RA_THRESHOLD. Exposed at module scope (not just inlined in
    run_sweep()) so the other experiments reuse the EXACT same M1
    configuration."""
    return M1Policy(kb_, config=M1Config(threshold=RA_THRESHOLD))


def _state_gain_only_factory(kb_):
    """M1 with the reasoning-gain term removed from the selection score
    (field_sim.communication.StateGainPolicy: score = state_gain, same
    threshold) - the field's own collective reasoning stays on."""
    return StateGainPolicy(StateGainConfig(threshold=RA_THRESHOLD))


# policy name -> (KnowledgeBase -> CommunicationPolicy factory [field_sim.
# experiments.harness.build_world()'s convention; None = broadcast],
# reasoning_enabled [collective-reasoning ablation - see
# run_roi_with_inference()'s own docstring]).
#   selective              - proposed method (M1: state gain + reasoning gain), field reasoning on
#   selective_no_reasoning - same M1 selection, field.reason() off
#   state_gain_only        - selection without the reasoning-gain term, field reasoning on
#   broadcast              - every candidate transmitted, field reasoning on
POLICIES = {
    "selective": (_m1_policy_factory, True),
    "selective_no_reasoning": (_m1_policy_factory, False),
    "state_gain_only": (_state_gain_only_factory, True),
    "broadcast": (None, True),
}


def _cfg(radius, seed):
    return ExperimentConfig(
        instances=INSTANCES, observation_radius=radius, seed=seed,
        evaporation_rate=EVAPORATION_RATE, change_prob=CHANGE_PROB,
        agent_move_step=AGENT_MOVE_STEP, steps=STEPS,
    )


def _job(args):
    policy, radius, seed = args
    factory, reasoning_enabled = POLICIES[policy]
    return policy, radius, seed, run_roi_with_inference(_cfg(radius, seed), factory, reasoning_enabled)


def run_sweep(frag_points):
    jobs = [(p, r, s) for p in POLICIES for _, r in frag_points for s in SEEDS]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        runs = {(p, r, s): res for p, r, s, res in pool.map(_job, jobs)}

    rows, data = [], {}
    for policy in POLICIES:
        for frag_name, radius in frag_points:
            res = [runs[(policy, radius, s)] for s in SEEDS]
            ccg = avg([r["ccg"] for r in res])
            cost = avg([r["cost"] for r in res])
            data[(policy, radius)] = (ccg, cost)
            rows.append({
                "policy": policy, "fragmentation": frag_name or "-", "radius": radius,
                "CCG": round(ccg, 4), "cost_deposits": round(cost, 1),
                "gain_per_1k_msgs": round(1000 * ccg / cost, 4) if cost else 0.0,
                "coverage_field": round(avg([r["ckc_field"] for r in res]), 4),
                "coverage_best_individual": round(avg([r["ckc_best"] for r in res]), 4),
                "truth_size": round(avg([r["window_mean"]["truth_size"] for r in res]), 2),
                "matched": round(avg([r["window_mean"]["matched"] for r in res]), 2),
                "missing": round(avg([r["window_mean"]["missing"] for r in res]), 2),
                "incorrect": round(avg([r["window_mean"]["incorrect"] for r in res]), 2),
            })
    return rows, data, runs


def validate(runs, frag_points) -> dict:
    """Same seed -> same environment and sensing under every policy (the
    candidate stream is policy-independent), no undefined coverage, and the
    reference closure never hit its iteration cap."""
    same_stream = all(
        len({runs[(p, r, s)]["candidate_stream_sha256"] for p in POLICIES}) == 1
        for _, r in frag_points for s in SEEDS
    )
    return {
        "candidate_stream_identical_across_policies": same_stream,
        "no_nan_coverage": all(not math.isnan(x["ckc_field"]) and not math.isnan(x["ckc_best"]) for x in runs.values()),
        "reference_iteration_limit_hits": sum(x["reference_iteration_limit_hits"] for x in runs.values()),
        "reference_single_valued_conflicts": sum(x["reference_conflicts"] for x in runs.values()),
        "field_single_valued_inference_conflicts": sum(x["field_inference_conflicts"] for x in runs.values()),
    }


def main(frag_points=FRAG):
    rows, data, runs = run_sweep(frag_points)
    radii = sorted({r for _, r in frag_points})
    checks = validate(runs, frag_points)

    print(f"=== CCG vs fragmentation ({len(INSTANCES)} instances, {len(SEEDS)} seeds) ===")
    print(f"{'policy':<23} {'radius':>7} {'CCG':>6} {'cov':>6} {'cost':>8}")
    for row in rows:
        print(f"{row['policy']:<23} {row['radius']:>7.0f} {row['CCG']:>6.3f} {row['coverage_field']:>6.3f} "
              f"{row['cost_deposits']:>8.0f}")
    print("validation:", checks)

    per_run = [
        {"policy": p, "radius": r, "seed": s, **{k: v for k, v in res.items() if k != "candidate_stream_sha256"}}
        for (p, r, s), res in sorted(runs.items(), key=lambda kv: (list(POLICIES).index(kv[0][0]), kv[0][1], kv[0][2]))
    ]
    write_csv("ccg_vs_fragmentation.csv", rows)
    write_json("ccg_vs_fragmentation_summary.json", {
        "seeds": SEEDS, "instances": INSTANCES, "radii": radii, "steady": STEADY,
        "coverage_definition": "|field triples & reference triples| / |reference triples|, "
                               "reference = closure of the current observable facts (field_sim.evaluation)",
        "rows": rows, "validation": checks, "per_run": per_run,
    })
    print("\nwrote results_main/ccg_vs_fragmentation.csv + _summary.json "
          "(draw the figure with: python figures/make_ccg_vs_fragmentation.py)")
    critical = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if critical:
        raise RuntimeError(f"validation failed: {critical}")


if __name__ == "__main__":
    main(frag_points=FRAG + EXTRA_RADII)
