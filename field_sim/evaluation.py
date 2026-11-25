"""Reference state and triple-based coverage, shared by every experiment.

Reference state (ground truth) at round t:
  the environment's CURRENT observable facts (one per (subject, relation),
  confidence 1.0) closed under the reasoning rules to a fixpoint
  (field_sim.reasoning.FieldReasoner.closure). Rebuilt from scratch every
  round - nothing from any field, past or present, enters it. Multi-valued
  relations keep every derived object (e.g. near's symmetric closure);
  single-valued conflicts are resolved and counted by the closure itself.

Coverage of an evaluated state (a field, or an individual's own memory):
  evaluated and reference are both turned into sets of (subject, relation,
  object) triples, and
      matched   = evaluated & truth
      coverage  = |matched| / |truth|
      missing   = |truth - evaluated|
      incorrect = |evaluated - truth|   (held, but not currently true)
  Coverage is undefined for an empty reference; score() returns NaN for it
  and the experiments refuse to aggregate one (the populations used never
  produce an empty reference - every observable relation always has a
  value).
"""

import math

from field_sim.field import edge_from_observation
from field_sim.types import FieldEdge, Observation


def reference_state(environment, kb, t: int, reasoner) -> dict[tuple, FieldEdge]:
    """Ground-truth edges at round t: the environment's current observable
    facts plus their reasoning closure."""
    gt_edges = {}
    for fact in environment.facts.values():
        obs = Observation(subject=fact.subject, relation=fact.relation, object=fact.object,
                          location=fact.location, obs_time=t, confidence=1.0, agent_id="ground_truth")
        gt_edges[obs.key] = edge_from_observation(obs, t)
    return reasoner.closure(gt_edges, kb, t)


def edge_triples(edges: dict[tuple, FieldEdge]) -> set[tuple[str, str, str]]:
    return {e.triple for e in edges.values()}


def memory_triples(memory: dict[tuple[str, str], str]) -> set[tuple[str, str, str]]:
    """An individual's own memory, kept as (subject, relation) -> latest
    directly observed object (the existing individual definition: direct
    observation only, latest value per observed fact), as triples."""
    return {(s, r, o) for (s, r), o in memory.items()}


def score(evaluated: set, truth: set) -> dict:
    matched = len(evaluated & truth)
    return {
        "coverage": matched / len(truth) if truth else math.nan,
        "matched": matched,
        "missing": len(truth) - matched,
        "incorrect": len(evaluated - truth),
        "truth_size": len(truth),
    }


def matched_by_relation(evaluated: set, truth: set) -> dict[str, tuple[int, int]]:
    """relation -> (matched, truth count), for per-relation coverage."""
    out: dict[str, list[int]] = {}
    for s, r, o in truth:
        cell = out.setdefault(r, [0, 0])
        cell[1] += 1
        if (s, r, o) in evaluated:
            cell[0] += 1
    return {r: (m, n) for r, (m, n) in out.items()}
