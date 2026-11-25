"""Heterogeneous representation and alias mapping: what the ontology's
grounding step (KnowledgeBase.map_to_candidate) contributes.

Existing mapping path (checked in code, see the report):
  - Agent.observe() builds a RawObservation from an environment fact with
    raw_relation = kb.encode(agent.modality, relation) - one fixed label per
    (modality, relation): "canonical" agents send a synonym (e.g.
    "located_in"), "numeric" a code ("101"), "abbrev" an abbreviation
    ("loc") - but subject and object are the CANONICAL instance ids taken
    straight from the fact. Entity heterogeneity therefore does not exist in
    the existing runs (this is the bypass); ontology.entity_aliases (the
    supported entity-alias mechanism, many-to-one, applied to subject and
    object) is empty.
  - map_to_candidate(): subject/object -> canonicalize_entity (entity_aliases),
    relation -> relation_aliases[modality] (unknown label -> rejected),
    domain/range checked against registered classes (an unregistered entity
    is accepted unchecked).
  - Reasoning rules match canonical relation names and their type
    constraints require a registered class - an alias entity never matches.

Conditions (same world, same agents, same sensor quality and movement, same
underlying facts observed every round - validated):
  common_on    every agent reports canonical relation names and canonical ids;
               grounding = map_to_candidate (modality "canonical")
  common_off   same raw reports; grounding without any alias table (labels
               kept verbatim; for a known relation the existing domain/range
               check still applies)
  hetero_on    vocabulary groups = the agent's existing modality (3 groups;
               agents i % 3): relation label = that modality's existing label;
               entity label = a per-group alias (canonical group: the canonical
               id; numeric group: "N###"; abbrev group: "e###"), stable for the
               run, one-to-one within a group (no two real entities share an
               alias). Grounding = map_to_candidate with the group aliases
               installed in ontology.entity_aliases for the run.
  hetero_off   same raw reports; grounding without alias tables: the trace is
               stored with its alias labels. A relation label that is not a
               known property has no schema, so no domain/range check is
               possible and it is stored untyped (its storage key is
               (subject, label) - an unknown label is not known to be
               multi-valued); an unknown entity is accepted unchecked (the
               existing policy). Nothing is converted to canonical form, so
               such traces never match a rule. How many of these traces a
               strict no-mapping grounding would reject (map_to_candidate's
               rule: a relation label that is not a known property is
               dropped) is counted separately, so a rejection effect is never
               confused with an alignment effect.

Evaluation only (never given to agents, policy, grounding or reasoning): the
known alias maps canonicalize every stored triple; duplicates collapse; then
coverage / precision against the reference closure. Storage and uploads keep
alias duplicates as they are.

Outputs: field_sim/experiments/results_additional/alias_mapping/
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

from field_sim import ontology
from field_sim.agent import sync_agent_locations
from field_sim.cardinality import is_multi_valued
from field_sim.evaluation import edge_triples, reference_state
from field_sim.experiments import exp_ccg_vs_fragmentation as ccg
from field_sim.experiments import exp_common as sync
from field_sim.experiments.harness import build_world
from field_sim.provenance import MULTI, ProvenanceRecorder
from field_sim.reasoning import FieldReasoner
from field_sim.types import Observation, RawObservation

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additional", "alias_mapping")
CONDITIONS = ["common_on", "common_off", "hetero_on", "hetero_off"]
RATES = sync.RATES
SEEDS = sync.SEEDS
FINAL_WINDOW = ccg.STEADY
INFERRED_RELATIONS = {r for r, src in ontology.property_source_type.items() if src == ontology.INFERRED}


class Scheme:
    """How each agent writes subjects, relations and objects."""

    def __init__(self, hetero: bool, instances: dict):
        self.hetero = hetero
        ids = sorted(instances)
        self.group_alias = {"canonical": {e: e for e in ids},
                            "numeric": {e: f"N{i:03d}" for i, e in enumerate(ids)},
                            "abbrev": {e: f"e{i:03d}" for i, e in enumerate(ids)}}

    def modality(self, agent) -> str:
        return agent.modality if self.hetero else "canonical"

    def relation_label(self, agent, kb, relation) -> str:
        return kb.encode(agent.modality, relation, "")[0] if self.hetero else relation

    def entity_label(self, agent, entity) -> str:
        return self.group_alias[agent.modality].get(entity, entity) if self.hetero else entity

    def entity_alias_map(self) -> dict:
        """alias -> canonical id over every group (used by mapping ON, and by the evaluator)."""
        return {a: e for g in self.group_alias.values() for e, a in g.items() if a != e} if self.hetero else {}


def observe(agent, env, kb, t, scheme):
    """Agent.observe() with the scheme's labels - identical fact selection,
    self-observation rule and reported quality."""
    raws, facts = [], []
    for fact in env.facts.values():
        if fact.subject != agent.agent_id and agent.location.distance_to(fact.location) > agent.observation_radius:
            continue
        if kb.encode(agent.modality, fact.relation, fact.object) is None:      # same availability rule as Agent.observe
            continue
        value = scheme.entity_label(agent, fact.object) if fact.relation in kb.object_properties else fact.object
        raws.append(RawObservation(subject=scheme.entity_label(agent, fact.subject), modality=scheme.modality(agent),
                                   raw_relation=scheme.relation_label(agent, kb, fact.relation), raw_value=value,
                                   location=fact.location, obs_time=t, observation_quality=agent.reliability,
                                   agent_id=agent.agent_id))
        facts.append(fact.triple)
    return raws, facts


def ground_without_mapping(raw, kb, t):
    """Mapping OFF: labels verbatim. (candidate or None, rejection reason or None)."""
    rel, subj, obj = raw.raw_relation, raw.subject, raw.raw_value
    if rel in kb.object_properties:
        spec = kb.object_properties[rel]
        sc, oc = kb.subject_class.get(subj), kb.subject_class.get(obj)
        if sc is not None and not kb.class_hierarchy.is_a(sc, spec.domain):
            return None, "domain"
        if oc is not None and not kb.class_hierarchy.is_a(oc, spec.range):
            return None, "range"
    elif rel in kb.datatype_properties:
        spec = kb.datatype_properties[rel]
        sc = kb.subject_class.get(subj)
        if sc is not None and not kb.class_hierarchy.is_a(sc, spec.domain):
            return None, "domain"
        try:
            spec.range(obj)
        except (TypeError, ValueError):
            return None, "datatype"
    return Observation(subject=subj, relation=rel, object=obj, location=raw.location, obs_time=t,
                       confidence=raw.observation_quality, agent_id=raw.agent_id), None


@contextmanager
def installed_entity_aliases(mapping: dict):
    saved = dict(ontology.entity_aliases)
    ontology.entity_aliases.clear()
    ontology.entity_aliases.update(mapping)
    try:
        yield
    finally:
        ontology.entity_aliases.clear()
        ontology.entity_aliases.update(saved)


class Evaluator:
    """Canonicalizes stored triples with the KNOWN alias maps - evaluation only."""

    def __init__(self, scheme: Scheme):
        self.ent = scheme.entity_alias_map() if scheme.hetero else {}
        self.rel = dict(ontology._alias_to_canonical)

    def canon(self, triple):
        s, r, o = triple
        return (self.ent.get(s, s), self.rel.get(r, r), self.ent.get(o, o))


def run_one(rate: float, condition: str, seed: int, steps: int = ccg.STEPS) -> dict:
    hetero, mapping = condition.startswith("hetero"), condition.endswith("_on")
    cfg = sync._cfg(rate, seed)
    cfg.steps = steps
    env, kb, agents, field, _ = build_world(cfg, policy_factory=None)
    scheme = Scheme(hetero, ccg.INSTANCES)
    m1 = ccg._m1_policy_factory(kb)
    rec = ProvenanceRecorder()
    field.provenance = rec
    ev = Evaluator(scheme)
    reasoner = FieldReasoner()
    facts_sha, stream_sha = hashlib.sha256(), hashlib.sha256()
    rows, rejections, existing_rejections = [], Counter(), Counter()
    alias_map = scheme.entity_alias_map() if mapping else {}
    with installed_entity_aliases(alias_map):
        assert mapping or not ontology.entity_aliases          # mapping OFF: no alias table anywhere
        for t in range(1, steps + 1):
            env.step(t)
            sync_agent_locations(agents, env)
            cand = {}
            for a in agents:
                raws, facts = observe(a, env, kb, t, scheme)
                facts_sha.update(repr((t, a.agent_id, facts)).encode())
                cs = []
                for raw in raws:
                    if mapping:
                        c = kb.map_to_candidate(raw, t)
                        if c is None:
                            rejections["mapping_rejected"] += 1
                    else:
                        c, why = ground_without_mapping(raw, kb, t)
                        if why:
                            rejections[why] += 1
                        # without an alias table, a label that is not a property name is unknown -
                        # map_to_candidate's rule (unknown relation -> reject) would drop it
                        if raw.raw_relation not in kb.object_properties and raw.raw_relation not in kb.datatype_properties:
                            existing_rejections["unknown_relation_label"] += 1
                    if c is not None:
                        cs.append(c)
                cand[a.agent_id] = cs
            stream_sha.update(repr([(t, ev.canon(c.triple), c.confidence, c.agent_id)
                                    for a in agents for c in cand[a.agent_id]]).encode())
            uploads = 0
            for a in agents:
                for c in cand[a.agent_id]:
                    if m1.evaluate(c, field, t, kb).transmit:
                        field.set(c, t)
                        uploads += 1
            field.decay(t)
            field.reason(kb, t)

            truth = edge_triples(reference_state(env, kb, t, reasoner))
            stored = [e.triple for e in field.edges.values()]
            canon = {ev.canon(x) for x in stored}
            q = sync.quality(canon, truth)
            inf_truth = {x for x in truth if x[1] in INFERRED_RELATIONS}
            inferred_edges = [e for e in field.edges.values() if e.fact_type == "inferred"]
            multi = sum(1 for e in inferred_edges if rec.classify(e).status == MULTI)
            rows.append({"t": t, **q, "uploads": uploads, "stored_edges": len(stored),
                         "alias_duplicates": len(stored) - len(canon),
                         "inferred_coverage": len(canon & inf_truth) / len(inf_truth) if inf_truth else math.nan,
                         "inferred_held": len(inferred_edges), "multi_agent_inferred": multi,
                         "canonical_field_sha": hashlib.sha256(repr(sorted(canon)).encode()).hexdigest()})
    return {"rounds": rows, "facts_sha": facts_sha.hexdigest(), "stream_sha": stream_sha.hexdigest(),
            "rejections": dict(rejections), "existing_grounding_would_reject": dict(existing_rejections)}


def _job(args):
    t0 = time.time()
    out = run_one(*args)
    out["seconds"] = time.time() - t0
    return args, out


def seed_rows(runs):
    out = []
    for (rate, cond, seed), o in sorted(runs.items()):
        R, last = o["rounds"], o["rounds"][-FINAL_WINDOW:]
        prec = [r["precision"] for r in last if not math.isnan(r["precision"])]
        ic = [r["inferred_coverage"] for r in last if not math.isnan(r["inferred_coverage"])]
        out.append({"rate": RATES[rate], "condition": cond, "seed": seed,
                    "uploads": sum(r["uploads"] for r in R), "uploads_round1": R[0]["uploads"],
                    "coverage_last20": statistics.mean(r["coverage"] for r in last),
                    "precision_last20": statistics.mean(prec) if prec else math.nan,
                    "inferred_coverage_last20": statistics.mean(ic) if ic else math.nan,
                    "stored_edges_last20": statistics.mean(r["stored_edges"] for r in last),
                    "alias_duplicates_last20": statistics.mean(r["alias_duplicates"] for r in last),
                    "inferred_held_last20": statistics.mean(r["inferred_held"] for r in last),
                    "multi_agent_inferred_last20": statistics.mean(r["multi_agent_inferred"] for r in last),
                    "multi_agent_inferred_fraction_last20": statistics.mean(
                        r["multi_agent_inferred"] / r["inferred_held"] for r in last if r["inferred_held"]) if any(
                        r["inferred_held"] for r in last) else math.nan,
                    **{f"rejected_{k}": v for k, v in o["rejections"].items()},
                    **{f"existing_grounding_would_reject_{k}": v for k, v in o["existing_grounding_would_reject"].items()},
                    "seconds": round(o["seconds"], 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[0.02])
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"config{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"conditions": CONDITIONS, "vocabulary_groups": 3,
                   "group_assignment": "agent modality (existing round-robin by sorted Humanoid id)",
                   "relation_labels_per_relation": "1 per group (existing modality label); common: canonical name",
                   "entity_labels_per_entity": "1 per group: canonical id / N### / e###; common: canonical id",
                   "datatype_values": "not aliased", "change_probs": args.rates, "radius": sync.RADIUS,
                   "seeds": SEEDS, "steps": ccg.STEPS, "policy": "M1 threshold 0.5, free live access",
                   "evaluation": "known alias maps canonicalize stored triples (evaluation only), dedup"}, f, indent=2)
    before = sync.manifest(sync.PRESERVED)
    jobs = [(r, c, s) for r in args.rates for c in CONDITIONS for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        runs = dict(ex.map(_job, jobs))
    rw = [{"rate": RATES[r], "condition": c, "seed": s, **{k: v for k, v in row.items() if k != "canonical_field_sha"}}
          for (r, c, s), o in sorted(runs.items()) for row in o["rounds"]]
    sync._write_csv(os.path.join(OUT_DIR, f"roundwise{args.tag}.csv"), rw)
    srows = seed_rows(runs)
    sync._write_csv(os.path.join(OUT_DIR, f"seed_summary{args.tag}.csv"), srows)
    by = defaultdict(list)
    for r in srows:
        by[(r["rate"], r["condition"])].append(r)
    keys = sorted({k for r in srows for k in r} - {"rate", "condition", "seed"})
    crows = []
    for (rate, cond), rs in sorted(by.items(), key=lambda kv: (kv[0][0], CONDITIONS.index(kv[0][1]))):
        row = {"rate": rate, "condition": cond, "seeds": len(rs)}
        for m in keys:
            row[f"{m}_mean"], row[f"{m}_sd"] = sync._ms([r.get(m, 0) for r in rs])
        crows.append(row)
    sync._write_csv(os.path.join(OUT_DIR, f"condition_summary{args.tag}.csv"), crows)
    checks = {
        "underlying_facts_identical_across_conditions": all(
            len({runs[(r, c, s)]["facts_sha"] for c in CONDITIONS}) == 1 for r in args.rates for s in SEEDS),
        "hetero_on_restores_common_on_exactly": all(
            runs[(r, "hetero_on", s)]["stream_sha"] == runs[(r, "common_on", s)]["stream_sha"]
            and [x["canonical_field_sha"] for x in runs[(r, "hetero_on", s)]["rounds"]]
            == [x["canonical_field_sha"] for x in runs[(r, "common_on", s)]["rounds"]]
            and [x["uploads"] for x in runs[(r, "hetero_on", s)]["rounds"]]
            == [x["uploads"] for x in runs[(r, "common_on", s)]["rounds"]] for r in args.rates for s in SEEDS),
        "common_off_equals_common_on": all(
            [x["canonical_field_sha"] for x in runs[(r, "common_off", s)]["rounds"]]
            == [x["canonical_field_sha"] for x in runs[(r, "common_on", s)]["rounds"]] for r in args.rates for s in SEEDS),
        "existing_results_unchanged": sync.manifest(sync.PRESERVED) == before}
    with open(os.path.join(OUT_DIR, f"summary{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"validation": checks, "elapsed_seconds": time.time() - t0}, f, indent=2)
    print("validation:", checks)
    for r in crows:
        print(f"{r['rate']:8s} {r['condition']:11s} up {r['uploads_mean']:6.0f}±{r['uploads_sd']:4.0f}"
              f" cov {r['coverage_last20_mean']:.3f}±{r['coverage_last20_sd']:.3f} prec {r['precision_last20_mean']:.3f}"
              f" infcov {r['inferred_coverage_last20_mean']:.3f} stored {r['stored_edges_last20_mean']:.0f}"
              f" dup {r['alias_duplicates_last20_mean']:.1f} multi {r['multi_agent_inferred_fraction_last20_mean']:.3f}"
              + "".join(f" {k[:-5]} {r[k]:.0f}" for k in r if (k.startswith("rejected_") or k.startswith("existing")) and k.endswith("_mean")))


if __name__ == "__main__":
    main()
