"""Selective deposition (restored): classify a candidate semantic trace
against the field's CURRENT belief for that key, independent of reasoning
value - "how much would writing this change what the field believes".

This classification existed before field_sim.communication.
ReasoningAwarePolicy (pure reasoning_gain) became the sole transmission
criterion, was removed, and has been reconstructed here at the user's
request to be layered together with reasoning_gain into
field_sim.communication.M1Policy - see that class's docstring for how the
two combine.

Reconstruction note: the field's write path stays field_sim.field.Field.set()
(a single unconditional upsert), unchanged - classification here only feeds
a transmission-decision SCORE, it does not change how a write is applied
once transmission is decided (no differentiated _add_novel/_reinforce/
_refresh/_overwrite write methods are being brought back). The original
mechanism also tracked competing claims across simultaneously disagreeing
agents (a conflict registry); that piece is intentionally NOT reconstructed
here - it is a separate, heavier concern from state_gain classification.
"""

from dataclasses import dataclass, field as dc_field

from field_sim.field import Field
from field_sim.types import Observation

NOVEL = "novel"            # field has no belief at all for this key yet
CORRECTION = "correction"  # field's current value disagrees, and either this candidate is MORE confident than it or that belief has gone stale
REINFORCE = "reinforce"    # same value, this candidate is more confident than the field currently holds
REFRESH = "refresh"        # same value, field's belief is old enough to be worth timestamping again
REDUNDANT = "redundant"    # same value (or a less-confident disagreement against a still-fresh belief) - nothing to gain by sending it


@dataclass
class SelectionConfig:
    # incoming confidence must exceed the field's by more than this to count
    # as REINFORCE/CORRECTION (per the design memo: "refresh when confidence
    # is higher than 0.2")
    reinforce_margin: float = 0.2
    correction_margin: float = 0.2   # same margin, applied to a value DISAGREEMENT instead of a same-value confirmation
    # field belief older than this (rounds) counts as REFRESH/CORRECTION,
    # not REDUNDANT (per the design memo: "refresh when time difference is
    # over 10")
    refresh_window: int = 10
    # NOVEL/CORRECTION rely on reasoning_gain to ever push a borderline case
    # over threshold; REINFORCE/REFRESH can't - the field's value is
    # unchanged in both cases, so reasoning_gain is structurally always 0.0
    # (the counterfactual inserts the same (subject,relation,object) already
    # in the field, so the reasoner's inferred-triple set can't change) -
    # verified via field_sim.communication.ReasoningAwarePolicy.evaluate(). With the
    # default threshold=0.5, state_gain alone therefore has to clear 0.5 for
    # REINFORCE/REFRESH to EVER transmit; below that they are permanently
    # unreachable regardless of any reasoning consideration.
    #
    # Values match the design memo's knowledge-gain scale: new edge(1.0) >
    # refresh-by-staleness(0.9) > refresh-by-confidence(0.8) > no change(0).
    # The memo's "compare metadata and update (recent time, higher
    # confidence)" rule applies uniformly whether or not the update also
    # changes the object value - CORRECTION (a value DISAGREEMENT that wins
    # on confidence or staleness) isn't a separate number in the memo's
    # list, so decide_deposition() below resolves its gain to whichever of
    # REINFORCE/REFRESH's numbers matches which condition actually won,
    # rather than a fixed constant of its own.
    state_gain: dict = dc_field(default_factory=lambda: {
        NOVEL: 1.0,
        REINFORCE: 0.8,
        REFRESH: 0.9,
        REDUNDANT: 0.0,
    })


@dataclass
class DepositionDecision:
    classification: str
    state_gain: float


def decide_deposition(obs: Observation, field: Field, t: int, config: SelectionConfig | None = None) -> DepositionDecision:
    """Classify `obs` against field.edges.get(obs.key) - the field's current
    belief, if any.

    obs.key is field_sim.cardinality.edge_key(): the (subject, relation)
    pair for a single-valued relation, so everything below applies exactly
    as before; the full triple for a multi-valued relation (near,
    potentialConflict, canInteractWith), so a stored edge with a DIFFERENT
    object for the same pair is not "the existing belief" at all - the
    candidate is NOVEL - and an existing edge, when found, always holds the
    same object (REINFORCE/REFRESH/REDUNDANT; the CORRECTION branch is
    unreachable for these relations).

    Per the design memo's knowledge-gain rule: a brand new edge (no existing
    belief for this key at all) is NOVEL (gain 1.0). Node existence
    (subject/object already known elsewhere in the field) doesn't change
    this: since edges are keyed by subject, a subject the field has never
    seen for this relation can never already have an edge here, so every
    "subject is new" case the memo lists (both node don't exist, only the
    object node exists) collapses to the same "no existing edge" check as
    the "subject already known" cases - the node-existence framing doesn't
    gate a different code path in a system with no separate node registry.

    When an edge for this key DOES exist, the memo's rule is "compare
    metadata and update (1. recent time, 2. higher confidence)" - applied
    the SAME way whether or not the update also changes the object value:
      - CORRECTION: value DISAGREES with the field's current belief, and
        wins on confidence (`correction_margin`) or staleness
        (`refresh_window`, measured from FieldEdge.value_since - see below)
      - REINFORCE: SAME value, wins on confidence (`reinforce_margin`)
      - REFRESH: SAME value, wins on staleness (`refresh_window`, measured
        from FieldEdge.last_deposition_time)
      - REDUNDANT (gain 0): neither confident nor stale enough to be worth
        sending, regardless of whether the value agrees or not.

    CORRECTION's gain isn't a constant of its own - it resolves to whichever
    of REINFORCE(0.8)/REFRESH(0.9) matches the condition that actually won,
    since the memo gives exactly two "update" numbers (confidence-based,
    staleness-based) and applies them uniformly to every "edge exists"
    update, not a third number for the value-disagreement case specifically.
    When BOTH conditions hold at once, the HIGHER of the two gains wins
    (REFRESH's 0.9, not REINFORCE's 0.8) - "use the highest applicable
    category," never sum them. This is computed as an actual max() over
    every condition that applies, not a first-match check order, so it
    stays correct even if state_gain's values are reconfigured later.

    The staleness exception (for CORRECTION) exists because a belief's
    stored `confidence` does not necessarily erode with time on its own -
    Field.decay() only lowers it when field.evaporation_rate > 0. At
    evaporation_rate == 0.0 a high-confidence belief that has since become
    wrong would otherwise be able to veto every later, lower-confidence
    correction FOREVER, since nothing ever brings its stored confidence
    down to let a new observation past the margin - an observed lock-in
    (verified: a wrong edge could stay stuck for the entire length of a
    run).

    CORRECTION's staleness check uses `value_since`, NOT
    `last_deposition_time`: a persistently repeated wrong value (the same
    glitch resubmitted every round) keeps passing as REFRESH and would keep
    `last_deposition_time` perpetually fresh, permanently blocking this
    staleness exception if it were keyed off that instead (verified: a
    wrong belief refreshed every round never lets a competing correction's
    gap reach refresh_window, no matter how long the run). `value_since`
    only advances when the value itself actually changes (Field.set()), so
    no amount of REFRESHing the SAME wrong value can prevent this clock
    from eventually granting a lower-confidence correction its chance."""
    cfg = config or SelectionConfig()
    existing = field.edges.get(obs.key)

    if existing is None:
        return DepositionDecision(NOVEL, cfg.state_gain[NOVEL])

    if existing.object != obs.object:
        candidates = []
        if obs.confidence > existing.confidence + cfg.correction_margin:
            candidates.append((CORRECTION, cfg.state_gain[REINFORCE]))
        if t - existing.value_since >= cfg.refresh_window:
            candidates.append((CORRECTION, cfg.state_gain[REFRESH]))
        # Self-report bypass: `obs` is this subject's own account of itself
        # (obs.subject == obs.agent_id, e.g. a Humanoid-linked Agent
        # reporting its own locatedAt) and isn't any less confident than
        # what's already believed. correction_margin/refresh_window exist to
        # arbitrate BETWEEN competing external observers of varying trust -
        # they don't apply here, since there's no competing source to
        # arbitrate against; the subject reporting on itself is authoritative
        # regardless of margin. Without this, a value that's genuinely
        # flip-flopping faster than correction_margin/refresh_window can
        # clear (e.g. an entity drifting back and forth across a region
        # boundary) gets stuck REDUNDANT every time the SAME sensor reports
        # its own new-but-not-more-confident state, and the field lags
        # behind information the subject already has about itself with zero
        # delay (verified: this was the dominant cause of the field
        # underperforming the best individual agent at low fragmentation in
        # exp_ccg_vs_fragmentation - see that experiment's notes).
        if obs.subject == obs.agent_id and obs.confidence >= existing.confidence:
            candidates.append((CORRECTION, cfg.state_gain[REFRESH]))
        if candidates:
            classification, gain = max(candidates, key=lambda c: c[1])
            return DepositionDecision(classification, gain)
        return DepositionDecision(REDUNDANT, cfg.state_gain[REDUNDANT])

    candidates = []
    if obs.confidence > existing.confidence + cfg.reinforce_margin:
        candidates.append((REINFORCE, cfg.state_gain[REINFORCE]))
    if t - existing.last_deposition_time >= cfg.refresh_window:
        candidates.append((REFRESH, cfg.state_gain[REFRESH]))
    if candidates:
        classification, gain = max(candidates, key=lambda c: c[1])
        return DepositionDecision(classification, gain)
    return DepositionDecision(REDUNDANT, cfg.state_gain[REDUNDANT])
