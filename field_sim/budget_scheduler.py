"""Per-round network-wide upload budget with sequential greedy selection.

A CONTROLLED SCHEDULING experiment, not a deployable protocol: every agent's
candidates for the round are pooled and compared centrally. Collecting the
candidates and running the scheduler cost nothing here - this measures what
the ranking signal does when uploads are scarce, not scheduling overhead.

Each round, at most `budget` candidates are uploaded:
  1. score every remaining candidate against the CURRENT collective state;
  2. among candidates whose score exceeds the threshold (0.5, as in M1),
     upload the highest score - ties broken by a seeded, policy-independent
     random priority per (seed, round, agent, triple), never by agent order;
  3. write it to the field immediately (Field.set), then rescore the rest;
  4. stop when the budget is used or no candidate is eligible - the budget is
     never filled with ineligible candidates.

Scores reuse the existing definitions unchanged:
  STATE_ONLY: state_gain (field_sim.selection.decide_deposition)
  COMBINED:   state_gain + reasoning_gain, unweighted (M1Policy's score),
              reasoning_gain = |I(F u {z}) \\ I(F)| exactly as
              field_sim.communication.ReasoningAwarePolicy computes it (the
              counterfactual is a copy - the real field is never touched;
              I(F) is computed once per step instead of once per candidate,
              which is the same value).

One exact shortcut: a candidate whose relation appears in no rule
antecedent cannot be a premise of anything, so inserting it can never make a
new inferred triple derivable - its reasoning gain is 0 without running the
closure (checked against ReasoningAwarePolicy in tests/test_budget_conflict).

Live collective reasoning (Field.reason) is NOT run here - the caller runs it
after the round's uploads and decay, exactly as in the unrestricted runs.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field as dc_field

from field_sim.field import Field, edge_from_observation
from field_sim.ontology import KnowledgeBase
from field_sim.reasoning import FieldReasoner
from field_sim.selection import SelectionConfig, decide_deposition

STATE_ONLY, COMBINED = "state_only", "combined"
THRESHOLD = 0.5


def tie_priority(seed: int, t: int, agent_id: str, triple: tuple) -> float:
    """Reproducible random priority in [0, 1), identical for both policies
    (depends only on seed, round and the candidate itself; no shared RNG
    state, so it cannot perturb the environment's RNG)."""
    h = hashlib.sha256(f"{seed}|{t}|{agent_id}|{'|'.join(triple)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def antecedent_relations(kb: KnowledgeBase) -> frozenset:
    return frozenset(atom.relation for rule in kb.reasoning_rules for atom in rule.antecedents)


def reasoning_gains(field: Field, candidates: list, indices, t: int, kb: KnowledgeBase,
                    reasoner: FieldReasoner, antecedents: frozenset) -> tuple[dict, dict]:
    """{i: reasoning gain}, {i: newly derivable inferred triples} for candidates[i],
    i in indices, against the current field - ReasoningAwarePolicy's DeltaU."""
    before = reasoner.reasoning_state(field.edges, kb, t)
    rg, enabled = {}, {}
    for i in indices:
        c = candidates[i]
        if c.relation not in antecedents:
            rg[i], enabled[i] = 0.0, set()
            continue
        counterfactual = dict(field.edges)
        counterfactual[c.key] = edge_from_observation(c, t)
        new = reasoner.reasoning_state(counterfactual, kb, t) - before
        rg[i], enabled[i] = float(len(new)), new
    return rg, enabled


@dataclass
class StepRecord:
    """One greedy step of a COMBINED round, with the state-only choice on the
    same state and candidate set (for 'did the reasoning gain change the pick')."""
    t: int
    step: int
    pick: int
    pick_relation: str
    pick_classification: str
    pick_state_gain: float
    pick_reasoning_gain: float
    enabled: Counter                    # relations of the inferred triples the pick newly makes derivable
    state_only_pick: int | None
    state_only_relation: str | None
    differs: bool


@dataclass
class RoundResult:
    uploaded: list[int] = dc_field(default_factory=list)    # candidate indices, in upload order
    steps: list[StepRecord] = dc_field(default_factory=list)


def select_round(field: Field, candidates: list, t: int, kb: KnowledgeBase, budget: int, mode: str,
                 priorities: list[float], reasoner: FieldReasoner | None = None,
                 selection_config: SelectionConfig | None = None, threshold: float = THRESHOLD,
                 record_steps: bool = False) -> RoundResult:
    """Upload at most `budget` of `candidates` (Observation list, agent_id set)
    to `field` by sequential greedy selection; see the module docstring."""
    cfg = selection_config or SelectionConfig()
    reasoner = reasoner or FieldReasoner()
    antecedents = antecedent_relations(kb)
    remaining = list(range(len(candidates)))
    result = RoundResult()

    while len(result.uploaded) < budget and remaining:
        decisions = {i: decide_deposition(candidates[i], field, t, cfg) for i in remaining}
        sg = {i: d.state_gain for i, d in decisions.items()}
        rg, enabled = {i: 0.0 for i in remaining}, {}
        if mode == COMBINED:
            rg, enabled = reasoning_gains(field, candidates, remaining, t, kb, reasoner, antecedents)
        elif mode != STATE_ONLY:
            raise ValueError(f"unknown mode {mode!r}")

        score = {i: sg[i] + rg[i] for i in remaining}
        eligible = [i for i in remaining if score[i] > threshold]
        if not eligible:
            break
        pick = max(eligible, key=lambda i: (score[i], priorities[i]))

        if record_steps and mode == COMBINED:
            so = [i for i in remaining if sg[i] > threshold]
            so_pick = max(so, key=lambda i: (sg[i], priorities[i])) if so else None
            result.steps.append(StepRecord(
                t=t, step=len(result.uploaded), pick=pick, pick_relation=candidates[pick].relation,
                pick_classification=decisions[pick].classification, pick_state_gain=sg[pick],
                pick_reasoning_gain=rg[pick], enabled=Counter(r for _, r, _ in enabled.get(pick, ())),
                state_only_pick=so_pick,
                state_only_relation=candidates[so_pick].relation if so_pick is not None else None,
                differs=so_pick != pick))

        field.set(candidates[pick], t)
        result.uploaded.append(pick)
        remaining.remove(pick)
    return result
