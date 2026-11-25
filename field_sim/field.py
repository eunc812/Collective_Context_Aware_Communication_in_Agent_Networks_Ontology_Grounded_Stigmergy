"""The shared field: the collective-cognition-aware communication
framework's "shared field" element - it gathers each agent's semantic trace
into the collective state.

Writing to the field is a single, unconditional operation: Field.set()
(last-write-wins upsert on the trace's storage key). Edges are keyed by
field_sim.cardinality.edge_key(): (subject, relation) for a single-valued
relation - a new object replaces the old one - and the full (subject,
relation, object) triple for a multi-valued relation (near,
potentialConflict, canInteractWith) - a different object is a separate
edge, and re-sending the same triple updates that edge's metadata only.
There is no novel/reinforce/refresh/correction
classification and no competing-claims/conflict-group tracking anywhere in
this module (or this codebase) - WHETHER a semantic trace is worth writing
is entirely M1's job (field_sim.communication.M1Policy), decided before
Field.set() is ever called; once M1 has decided to transmit, the field's
only remaining job is to record it.

Field.decay() is the framework's "temporal decay" element: at the end of
every round, every edge's confidence is multiplied by (1 - evaporation_rate)
and edges that fall below min_confidence are dropped.

Reasoning (M2) is a separate layer built on top of this one - see
field_sim.reasoning.FieldReasoner (the pure fixpoint reasoner) and
Field.reason() below (the live, one-hop-per-round counterpart).
"""

from collections import Counter

from field_sim.ontology import KnowledgeBase
from field_sim.types import FieldEdge, Location, Observation

NOVEL = "novel"  # kept only as Field.reason()'s "this triple didn't exist before this round" marker - see reason()


def pick_conflict_winner(objects_with_confidence: list[tuple[str, float]]) -> str:
    """The explicit rule for reasoning conclusions that disagree on a
    SINGLE-valued relation (several objects derived for the same subject
    and relation): the object with the highest confidence wins, ties go to
    the lexicographically smallest object. Depends only on the set of
    derivations, never on rule or binding order. Shared by Field.reason()
    and field_sim.reasoning.FieldReasoner.closure()."""
    return min(objects_with_confidence, key=lambda oc: (-oc[1], oc[0]))[0]


def edge_from_observation(obs: Observation, t: int, fact_type: str = "observed") -> FieldEdge:
    """Build the FieldEdge a fresh write of `obs` would create - shared by
    Field.set() (the real, mutating write path) and field_sim.reasoning.
    FieldReasoner (the pure, hypothetical-insertion path M1 uses for its
    counterfactual Reason(F u {z_i}))."""
    return FieldEdge(
        subject=obs.subject,
        relation=obs.relation,
        object=obs.object,
        location=obs.location,
        confidence=obs.confidence,
        last_observation_time=obs.obs_time,
        last_deposition_time=t,
        value_since=t,
        support_count=1,
        sources={obs.agent_id},
        fact_type=fact_type,
    )


class Field:
    def __init__(self, evaporation_rate: float = 0.05, min_confidence: float = 0.05):
        self.edges: dict[tuple, FieldEdge] = {}
        self.evaporation_rate = evaporation_rate
        self.min_confidence = min_confidence
        self.write_log: list[tuple[int, str, tuple]] = []
        # Diagnostics only, never read by any decision:
        #   replacements["observed->inferred"] etc. - a single-valued edge
        #     whose OBJECT was replaced, by the fact_type of old -> new writer.
        #   inference_conflicts - (t, pair, objects, chosen) whenever one
        #     reason() pass derived several different objects for the same
        #     single-valued pair (see reason()).
        self.replacements: Counter = Counter()
        self.inference_conflicts: list[tuple] = []
        # Optional analysis-only recorder (field_sim.provenance.ProvenanceRecorder):
        # notified of every write and of each reasoning pass's firings, never read
        # back by the field or any policy. None (the default) = no recording.
        self.provenance = None

    def set(self, obs: Observation, t: int, fact_type: str = "observed") -> str:
        """The field's only write path: unconditionally record `obs` as the
        current belief for (subject, relation) - a fresh edge if the key is
        new, an in-place overwrite (object/confidence/timestamps/sources)
        if it already exists. No comparison against the existing value
        influences WHETHER this happens - that decision was already made by
        M1 before this was ever called; Field.set() only decides HOW to
        represent the write, which is field infrastructure, not a
        communication-relevance judgment. Two exceptions, both bookkeeping
        rather than a relevance judgment:

          - FieldEdge.value_since only advances when `obs.object` differs
            from what was already there (see its own docstring).
          - FieldEdge.fact_type does NOT get demoted from "inferred" to
            "observed" when the value is unchanged (existing.object ==
            obs.object) - a direct observation that merely reconfirms what
            reasoning already concluded isn't evidence against that
            inference (its premises haven't been retracted), so it
            shouldn't erase the fact that this triple is - still, also -
            an inferred conclusion. This is the actual enforcement of the
            invariant FieldEdge.fact_type's docstring has always described
            ("a later write to the same key never demotes this") - until
            this check existed, every observed re-confirmation overwrote
            fact_type unconditionally, silently demoting it anyway (see
            field_sim.communication.ReasoningAwarePolicy's counterfactual,
            which mirrors this exact rule for the same reason: reasoning_
            utility_count() counts CURRENT fact_type=="inferred" triples,
            so that silent demotion was read back as a loss of reasoning
            value - producing a negative reasoning_gain for a same-value
            REINFORCE/REFRESH candidate that should have scored exactly 0).
            A genuine VALUE change (a correction) still supersedes the
            inference with a fresh fact_type, same as before.

        For a multi-valued relation the key includes the object, so a write
        never changes an existing edge's object: a different object lands on
        its own edge, and the same triple only refreshes metadata.

        Returns "new" or "updated" (write_log/metrics bookkeeping only)."""
        existing = self.edges.get(obs.key)
        if existing is None:
            self.edges[obs.key] = edge_from_observation(obs, t, fact_type)
            outcome = "new"
        else:
            value_changed = existing.object != obs.object
            if value_changed:
                existing.value_since = t
                self.replacements[f"{existing.fact_type}->{fact_type}"] += 1
            existing.object = obs.object
            existing.confidence = obs.confidence
            existing.last_observation_time = obs.obs_time
            existing.last_deposition_time = t
            existing.support_count += 1
            existing.sources.add(obs.agent_id)
            if value_changed or existing.fact_type != "inferred":
                existing.fact_type = fact_type
            outcome = "updated"
        self.write_log.append((t, outcome, obs.key))
        if self.provenance is not None:
            self.provenance.on_set(obs, t, fact_type)
        return outcome

    def decay(self, t: int) -> list[tuple[str, str]]:
        """Evaporate every edge's confidence one round; drop edges that
        fall below min_confidence. Returns the keys removed this round."""
        removed = []
        for key, edge in list(self.edges.items()):
            edge.confidence *= 1.0 - self.evaporation_rate
            if edge.confidence < self.min_confidence:
                del self.edges[key]
                removed.append(key)
        return removed

    def reason(self, kb: KnowledgeBase, t: int) -> list[tuple[str, tuple[str, str]]]:
        """M2, live/mutating path: ask the knowledge base which reasoning
        rules (field_sim.ontology.ReasoningRule - this domain's sole
        reasoning mechanism, M2's general variable join) fire against the
        field's current edges, and write each firing through Field.set(),
        same as any other semantic trace. No agent sends these; they're
        synthesized from what the field already collectively holds - new
        cognition emerging from the COMPOSITION of existing cognition, per
        the framework's reasoning element.

        One pass per call, not a fixpoint (contrast field_sim.reasoning.
        FieldReasoner, the pure/hypothetical-insertion counterpart used by
        M1's counterfactual and by ground-truth computation): a rule chain
        (one rule's consequent feeding another rule's antecedent) resolves
        one hop per round here, letting reasoning change incrementally over
        time instead of all at once.

        No retraction: an inferred edge that stops being re-derivable just
        sits there (still subject to the same Field.decay() as everything
        else) rather than being deleted the moment its premises stop
        holding. A same-round "retract anything not currently re-derivable,
        cascading to whatever depended on it" pass was tried and reverted
        (2026-09-09) - it did shrink the number of stale inferred beliefs,
        but net INCREASED field error at low fragmentation, because it
        can't tell "the world's antecedents genuinely changed" apart from
        "the field's OWN copy of a premise is momentarily missing/decayed
        even though the premise is still true" - it deleted correct,
        still-valid inferences whenever the field's imperfect memory of
        their premises lapsed, which cost more than the stale-belief
        cleanup saved. Revisit with a design that only retracts on a
        premise CONTRADICTION (a still-present premise edge whose value
        changed) rather than on mere premise ABSENCE, if this is worth
        another attempt.

        Returns a list of ("novel" | "updated", key) per firing - "novel"
        marks a triple that did not exist in the field before this call;
        a firing that only refreshes an already-known inferred triple is
        marked "updated".

        Per the design memo, an inferred edge's metadata is (same area,
        newest time, highest confidence among conditions, condition
        sources) - the first three are already how Observation.confidence/
        obs_time/location get built for a firing (field_sim.ontology.
        KnowledgeBase.infer_with_provenance()); the union of every
        antecedent edge's OWN agent sources - not just a single synthetic
        "inference:<rule>" label - is filled into FieldEdge.sources here,
        since only this method sees the antecedent FieldEdges (Field.set()
        only ever sees the single already-built candidate). FieldEdge.
        rule_id/condition_sources (which rule, which premise EDGES - not
        agent ids) are filled in the same way, for the same reason, and
        reflect only this, the most recent pass that wrote this edge (see
        FieldEdge's docstring for why that's different from `sources`, which
        accumulates across every firing an edge has ever had).

        Snapshot semantics: every rule is evaluated against the edges as
        they stood when this call started (and every premise's sources/key
        is read before anything is written), so a conclusion written in this
        pass is never reused as a premise in the same pass.

        One write per distinct conclusion per pass. When several firings
        (different rules, or different bindings of one rule) derive the
        same storage key, they are merged before writing:
          - confidence = max over the firings, obs_time = max (the design
            memo's "highest confidence / newest time among conditions",
            applied across firings as well as across premises);
          - location = that of the highest-confidence firing (ties broken
            by rule name, then premise keys - deterministic);
          - sources = union of every firing's premise sources,
            condition_sources = union of every firing's premise keys,
            rule_id = the sorted, comma-joined names of every rule that
            fired for it.
        For a multi-valued relation the key includes the object, so merged
        firings always agree on it. For a single-valued relation, firings
        for the same pair can disagree on the object; that is a conflict,
        resolved by pick_conflict_winner() (highest confidence, then the
        lexicographically smallest object - independent of rule order) and
        recorded in self.inference_conflicts rather than decided silently
        by whichever firing happened to be written last."""
        groups: dict[tuple, list] = {}
        for candidate, rule, premises in kb.infer_with_provenance(self.edges):
            groups.setdefault(candidate.key, []).append((
                candidate, rule.name,
                set().union(*(p.sources for p in premises)),
                {p.key for p in premises},
            ))

        writes = []
        for key, firings in groups.items():
            objects = {c.object for c, _, _, _ in firings}
            if len(objects) > 1:
                chosen = pick_conflict_winner([(c.object, c.confidence) for c, _, _, _ in firings])
                self.inference_conflicts.append((t, key, tuple(sorted(objects)), chosen))
                firings = [f for f in firings if f[0].object == chosen]
            best = min(firings, key=lambda f: (-f[0].confidence, f[1], sorted(map(repr, f[3]))))[0]
            merged = Observation(
                subject=best.subject, relation=best.relation, object=best.object, location=best.location,
                obs_time=max(c.obs_time for c, _, _, _ in firings),
                confidence=max(c.confidence for c, _, _, _ in firings),
                agent_id=f"inference:{','.join(sorted({r for _, r, _, _ in firings}))}",
            )
            writes.append((merged, firings))

        if self.provenance is not None:
            # before any write, so premise keys resolve to their snapshot versions
            self.provenance.begin_inference({
                merged.key: [(rule, tuple(sorted(keys, key=repr))) for _, rule, _, keys in firings]
                for merged, firings in writes
            })

        results = []
        for merged, firings in writes:
            is_new = merged.key not in self.edges
            outcome = self.set(merged, t, fact_type="inferred")
            edge = self.edges[merged.key]
            # Field.set() always seeds sources from candidate.agent_id (here
            # the synthetic "inference:<rules>" placeholder, since an
            # inferred candidate isn't from any one agent) - drop it in
            # favor of the real agent sources, still unioned with whatever
            # accumulated from earlier passes over this same edge.
            edge.sources.discard(merged.agent_id)
            for _, _, agent_sources, _ in firings:
                edge.sources |= agent_sources
            edge.rule_id = ",".join(sorted({r for _, r, _, _ in firings}))
            edge.condition_sources = set().union(*(keys for _, _, _, keys in firings))
            self.write_log.append((t, f"inferred_{outcome}", merged.key))
            results.append((NOVEL if is_new else "updated", merged.key))
        return results

    def query_near(self, location: Location, radius: float) -> list[FieldEdge]:
        return [e for e in self.edges.values() if e.location.distance_to(location) <= radius]

    def query(self, subject: str | None = None, relation: str | None = None) -> list[FieldEdge]:
        return [
            e
            for e in self.edges.values()
            if (subject is None or e.subject == subject)
            and (relation is None or e.relation == relation)
        ]
