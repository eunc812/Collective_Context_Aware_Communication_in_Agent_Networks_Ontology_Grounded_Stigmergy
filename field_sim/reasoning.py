"""M2: field-level ontology / semantic reasoning.

Pure, side-effect-free counterpart to field_sim.field.Field.reason(): where
Field.reason() mutates the LIVE shared field once per simulation round (a
single unconditional Field.set() upsert per firing, logged to write_log -
the real write history), FieldReasoner.closure() takes a snapshot of facts
and returns the fixpoint materialization of that snapshot (repeated
reasoning-rule application until nothing new fires), touching nothing else -
not the input dict, not any FieldEdge inside
it. That purity is what makes it usable for:

  - M1's reasoning-gain term (field_sim.communication.ReasoningAwarePolicy,
    nested inside M1Policy): compare Reason(F) against Reason(F u {z_i})
    for a candidate observation that was never actually written anywhere -
    a counterfactual Field.reason() can't do, since it only ever advances
    the one real field it's attached to.
  - ground truth / method comparisons: Reason(F_GT) vs Reason(F_method) over
    field snapshots that were never live Fields.

F_0 = the input facts, taken as given (their existing fact_type - "observed"
or "inferred" - is preserved as-is; this module never relabels an input
fact). Each iteration evaluates every rule in KnowledgeBase.reasoning_rules
(this domain's sole reasoning mechanism, the general variable join -
field_sim.ontology.ReasoningRule) against the facts accumulated so far
(one snapshot per iteration), and adds every genuinely new storage key
(field_sim.cardinality.edge_key). This repeats until an iteration adds no
new triple - i.e. materialization to a fixpoint. That means a multi-hop
chain (one rule's consequent feeding another rule's antecedent) resolves
within a single closure() call, instead of spreading across simulation
rounds the way Field.reason() does. Reaching max_iterations while triples
are still being added is recorded (self.stats["iteration_limit_hits"]),
never a silent stop.

Multi-valued relations (near, potentialConflict, canInteractWith) are keyed
by the full triple, so a derivation with a different object than an
existing entry is simply another triple - all of them are kept, and the
result does not depend on rule order.

Single-valued relations keep one object per (subject, relation). A
derivation that disagrees with what is already there is a conflict, handled
explicitly and counted (self.stats["single_valued_conflicts"], per relation
in self.conflicts_by_relation):
  - an existing entry wins over a later derivation: an input fact (an
    observation of the environment) is the authority on its own relation,
    and an inferred entry from an earlier iteration needed fewer rule
    applications than the challenger;
  - several different objects derived for the same NEW pair in the same
    iteration are resolved by field_sim.field.pick_conflict_winner()
    (highest confidence, then smallest object) - independent of rule order.
No current rule derives an observable single-valued relation, and the
inferred single-valued ones are functions of single-valued premises, so in
this domain these counters are expected to stay at 0; they are recorded so
a later rule change cannot introduce silent first-wins behavior again.

This module answers "is this fact entailed?", not "what should the live
field currently believe?" - Field.reason() still owns the latter for the
real, written field.
"""

from collections import Counter

from field_sim.cardinality import is_multi_valued
from field_sim.field import edge_from_observation, pick_conflict_winner
from field_sim.ontology import KnowledgeBase
from field_sim.types import FieldEdge, Observation

DEFAULT_MAX_ITERATIONS = 20


class FieldReasoner:
    """M2. Holds no facts of its own, just orchestrates fixpoint
    materialization over whatever fact snapshot it's handed. `stats` and
    `conflicts_by_relation` accumulate diagnostics across calls (callers
    that want per-call numbers reset or diff them)."""

    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self.max_iterations = max_iterations
        self.stats: Counter = Counter()
        self.conflicts_by_relation: Counter = Counter()

    def closure(
        self, edges: dict[tuple, FieldEdge], kb: KnowledgeBase, t: int
    ) -> dict[tuple, FieldEdge]:
        """Return a NEW facts dict: `edges` plus every fact derivable from it
        by repeated rule application, to a fixpoint. Never mutates `edges`
        or the FieldEdge objects inside it."""
        facts = dict(edges)
        rejected: set[tuple] = set()  # (key, losing object) - each conflict counted once per call
        self.stats["closures"] += 1
        for _ in range(self.max_iterations):
            new: dict[tuple, list[Observation]] = {}
            for candidate in kb.infer(facts):
                existing = facts.get(candidate.key)
                if existing is None:
                    new.setdefault(candidate.key, []).append(candidate)
                elif existing.object != candidate.object:  # single-valued only: the key would differ otherwise
                    self._conflict(candidate, rejected)
            if not new:
                break
            for key, derivations in new.items():
                if len({c.object for c in derivations}) > 1:
                    chosen = pick_conflict_winner([(c.object, c.confidence) for c in derivations])
                    for c in derivations:
                        if c.object != chosen:
                            self._conflict(c, rejected)
                    derivations = [c for c in derivations if c.object == chosen]
                facts[key] = _inferred_edge(_merge(derivations), t)
        else:
            self.stats["iteration_limit_hits"] += 1
        return facts

    def _conflict(self, losing: Observation, rejected: set) -> None:
        assert not is_multi_valued(losing.relation), losing.relation
        if (losing.key, losing.object) in rejected:
            return
        rejected.add((losing.key, losing.object))
        self.stats["single_valued_conflicts"] += 1
        self.conflicts_by_relation[losing.relation] += 1

    def reasoning_state(
        self, edges: dict[tuple, FieldEdge], kb: KnowledgeBase, t: int
    ) -> set[tuple[str, str, str]]:
        """I(F): the set of (subject, relation, object) triples this fact set
        entails that are inferred rather than directly observed - what a
        future field_sim.metrics compares across communication methods
        (reasoning recall/precision)."""
        facts = self.closure(edges, kb, t)
        return {(e.subject, e.relation, e.object) for e in facts.values() if e.fact_type == "inferred"}


Triple = tuple[str, str, str]


def reasoning_utility_count(inferred: set[Triple]) -> float:
    """U(I) = |I| - the raw number of inferred triples, unweighted. A plain
    descriptive count (e.g. for reporting how much a fact set currently
    entails) - field_sim.communication.ReasoningAwarePolicy does NOT use
    this for its own DeltaU any more (DeltaU_i is a set-difference SIZE,
    |R(F_t u {z_i}) \\ R(F_t)|, not U(R(F_t u {z_i})) - U(R(F_t)): a plain
    count difference isn't monotonic once z_i's insertion can overwrite an
    existing edge rather than purely add one, which is exactly the case
    that formula needs to stay non-negative - see that class's own
    docstring)."""
    return float(len(inferred))


def _merge(derivations: list[Observation]) -> Observation:
    """Several derivations of the SAME triple in one iteration -> one
    candidate: highest confidence, newest time, location of the
    highest-confidence derivation (ties: first in rule order - location
    plays no part in the triple, so this cannot change the closure's
    contents)."""
    best = max(derivations, key=lambda c: c.confidence)
    if len(derivations) == 1:
        return best
    return Observation(subject=best.subject, relation=best.relation, object=best.object, location=best.location,
                       obs_time=max(c.obs_time for c in derivations), confidence=best.confidence,
                       agent_id=best.agent_id)


def _inferred_edge(candidate: Observation, t: int) -> FieldEdge:
    return edge_from_observation(candidate, t, fact_type="inferred")
