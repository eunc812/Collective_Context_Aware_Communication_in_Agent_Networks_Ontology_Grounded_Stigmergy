"""Communication policies: sender-side decisions about whether an agent's
semantic trace gets transmitted to the shared field this round.

Per the collective-cognition-aware communication framework:

  M1 - reasoning-aware communication (M1Policy). Each agent selectively
       transmits according to BOTH the state-gain classification (field_sim.
       selection.decide_deposition's NOVEL/CORRECTION/REINFORCE/REFRESH/
       REDUNDANT -> state_gain: "how new/confident/stale is this belief")
       AND the REASONING impact its semantic trace would have on the
       field's collective cognition (reasoning_gain/DeltaU, via the
       narrower ReasoningAwarePolicy nested inside M1Policy) - M1 itself IS
       this combination, not reasoning_gain alone: score = state_gain +
       DeltaU, transmit iff score > threshold. field_sim.field.Field itself
       has exactly one write path (Field.set(), unconditional upsert/
       last-write-wins) - state_gain/reasoning_gain only ever feed this
       transmission-decision score, never a differentiated write.

  BroadcastPolicy - every candidate is transmitted, unconditionally - the
       reasoning ground truth used throughout field_sim.metrics.

  ReasoningAwarePolicy - the narrower, reasoning-gain-ONLY policy nested
       inside M1Policy (M1Config.reasoning_config configures it): "if this
       local cognition were communicated now, how much would it change the
       current collective reasoning state?" - and nothing else, no state
       gain. Deadlocks whenever no single candidate has reasoning value in
       isolation (see ReasoningAwareConfig's docstring) - M1Policy avoids
       that by having state_gain justify a first-time or
       contradicting observation on its own, regardless of reasoning value.
       ReasoningAwarePolicy remains usable standalone (reasoning-only
       baseline), independent of M1.

  StateGainPolicy - the complementary state-gain-ONLY policy: M1's other
       half in isolation. score = state_gain (field_sim.selection.
       decide_deposition's classification), transmit iff score > threshold
       - no reasoning value considered at all, same way ReasoningAwarePolicy
       considers no state value at all. Unlike ReasoningAwarePolicy, this
       one never deadlocks on a first-time observation (NOVEL's state_gain
       alone already clears a typical threshold) - its own failure mode is
       the opposite: once a value has been sent once, further REINFORCE/
       REFRESH of it depends only on confidence/staleness, never on whether
       resending it would unlock new reasoning.
"""

from dataclasses import dataclass, field as dc_field

from field_sim import selection
from field_sim.field import Field, edge_from_observation
from field_sim.ontology import KnowledgeBase
from field_sim.reasoning import FieldReasoner
from field_sim.selection import SelectionConfig, decide_deposition
from field_sim.types import Observation


@dataclass
class PolicyDecision:
    transmit: bool
    score: float
    reasoning_gain: float = 0.0
    novelty: float = 0.0  # 1.0 iff state_decision.classification == NOVEL (M1Policy only; ReasoningAwarePolicy never sets this), else 0.0
    classification: str = ""  # field_sim.selection classification (M1Policy only), else ""


class CommunicationPolicy:
    """Sender-side: does candidate `obs` get transmitted to `field` this round?"""

    def evaluate(self, obs: Observation, field: Field, t: int, kb: KnowledgeBase | None = None) -> PolicyDecision:
        raise NotImplementedError


class BroadcastPolicy(CommunicationPolicy):
    """Every observation is transmitted - the reasoning ground truth
    (field_sim.metrics) is what running this policy produces."""

    def evaluate(self, obs: Observation, field: Field, t: int, kb: KnowledgeBase | None = None) -> PolicyDecision:
        return PolicyDecision(transmit=True, score=1.0)


@dataclass
class ReasoningAwareConfig:
    """Configuration for ReasoningAwarePolicy (the reasoning-gain-ONLY
    policy nested inside M1Policy).

    threshold: score = DeltaU; transmit iff score > threshold. No novelty
      term, no message-cost penalty - reasoning value is the ONLY input to
      this score, by design (transmission cost isn't weighed against it
      here either; field_sim.experiments.exp_roi measures cost separately,
      as total deposits sent, not through this score).

    A direct, intended consequence of that purity: ANY rule needing 2+
    antecedents is a permanent, order-independent deadlock the moment no
    single antecedent has reasoning value on its own - every candidate
    scores DeltaU=0 forever, so nothing is ever transmitted and the rule
    never fires (verified repeatedly - see field_sim.experiments.exp_roi/
    exp_persist, which measure a general multi-subject domain where almost
    nothing individually completes a rule; ReasoningAwarePolicy transmits
    literally nothing there, and tests/test_core.py's
    test_joint_suppression_deadlock_when_no_premise_has_individual_value
    checks exactly this). That is not a bug this class works around - it's
    the honest behavior of reasoning-gain-alone communication, and the
    whole reason M1Policy exists: cold start is solved there, by layering
    state_gain (field_sim.selection.decide_deposition's NOVEL case) on top,
    never by diluting ReasoningAwarePolicy's own score with anything other
    than DeltaU.
    """

    threshold: float = 0.5
    max_reasoner_iterations: int = 20


class ReasoningAwarePolicy(CommunicationPolicy):
    """Reasoning-gain-ONLY communication: "if this local cognition were
    communicated now, how much would it change the current collective
    reasoning state?" - and nothing else. The narrower policy M1Policy
    nests inside itself for its DeltaU term; also usable standalone as a
    reasoning-only baseline.

    Let F_t be the field's current explicit edges. For candidate z_i:
        I_t      = R(F_t)              (FieldReasoner.reasoning_state)
        I_t_plus = R(F_t u {z_i})      (same reasoner, counterfactual insert)
        DeltaU_i = |I_t_plus \\ I_t|     (count of triples NEWLY unlocked by
                                        z_i - a set difference, not a count
                                        difference)
        score    = DeltaU_i
        transmit iff score > threshold

    DeltaU_i is a set-difference SIZE, not U(I_t_plus) - U(I_t) (a raw count
    difference) - deliberately, so it is non-negative BY CONSTRUCTION,
    regardless of what happens to any triple already in I_t. This matters
    because the counterfactual insertion below is not always purely
    additive: when obs.key already has an edge, inserting z_i OVERWRITES it
    for this evaluate() call, which can cause I_t_plus to lose some triple
    I_t had (e.g. a longer reasoning chain field.edges' own closure already
    credited at that key, which z_i's direct value now occupies instead -
    reasoning_state() takes a full multi-hop closure, not just field.edges'
    literal, one-hop-per-round contents, so it can count triples that
    haven't actually been materialized by Field.reason() yet). A plain count
    difference would score that loss as negative reasoning_gain - the
    opposite of what "how much does z_i ADD" is supposed to mean, and
    something field_sim.reasoning.reasoning_utility_count's own docstring
    says should never happen ("DeltaU_i >= 0 always for a purely-additive
    candidate" - true only because this formula no longer relies on z_i's
    insertion actually BEING purely additive).

    The counterfactual insertion (`counterfactual = dict(field.edges); ...`)
    builds a NEW dict and never mutates field.edges or any FieldEdge inside
    it - the real field is untouched by evaluate(). It is keyed by
    obs.key (field_sim.cardinality.edge_key): for a single-valued relation
    z_i replaces the stored object as described above; for a multi-valued
    relation it is added NEXT TO any other object of the same pair, so the
    insertion is purely additive and DeltaU counts exactly the inferred
    triples z_i newly makes derivable (each once - I_t/I_t_plus are sets).
    """

    def __init__(self, kb: KnowledgeBase, reasoner: FieldReasoner | None = None, config: "ReasoningAwareConfig | None" = None):
        self.kb = kb
        self.config = config or ReasoningAwareConfig()
        self.reasoner = reasoner or FieldReasoner(max_iterations=self.config.max_reasoner_iterations)

    def evaluate(self, obs: Observation, field: Field, t: int, kb: KnowledgeBase | None = None) -> PolicyDecision:
        kb = kb or self.kb
        cfg = self.config

        before = self.reasoner.reasoning_state(field.edges, kb, t)
        counterfactual = dict(field.edges)
        counterfactual[obs.key] = edge_from_observation(obs, t)
        after = self.reasoner.reasoning_state(counterfactual, kb, t)

        # Only triples NEWLY unlocked by z_i count (set difference) - see
        # this class's docstring for why that, not a raw count difference,
        # is what keeps DeltaU non-negative by construction regardless of
        # what z_i's insertion causes to disappear from the counterfactual's
        # inferred set. (This used to need a fact_type-preservation special
        # case here, matching Field.set()'s own rule, to stop a same-value
        # reconfirmation of an existing INFERRED edge from scoring a spurious
        # -1 under the old count-difference formula - moot now: that case
        # nets to 0 newly-unlocked triples either way, never a loss.)
        delta_u = float(len(after - before))
        score = delta_u

        return PolicyDecision(
            transmit=score > cfg.threshold, score=score, reasoning_gain=delta_u,
        )


@dataclass
class StateGainConfig:
    """Configuration for StateGainPolicy (the state-gain-ONLY policy - M1's
    other half in isolation, complementary to ReasoningAwareConfig).

    threshold: score = state_gain; transmit iff score > threshold. No
      reasoning term at all - state_gain is the ONLY input to this score,
      mirroring ReasoningAwarePolicy's purity in the opposite direction.
    """

    selection_config: SelectionConfig = dc_field(default_factory=SelectionConfig)
    threshold: float = 0.5


class StateGainPolicy(CommunicationPolicy):
    """State-gain-ONLY communication: "how new/confident/stale is this
    belief, independent of what it would do to collective reasoning?" -
    and nothing else. field_sim.selection.decide_deposition's NOVEL/
    CORRECTION/REINFORCE/REFRESH/REDUNDANT classification is the ONLY
    input; M1Policy's other term (ReasoningAwarePolicy's DeltaU) never
    enters this score at all.

        score = state_gain
        transmit iff score > threshold

    Never touches the field or any reasoning machinery - decide_deposition
    only reads field.edges to compare against the candidate, same read-only
    discipline every other policy here follows.
    """

    def __init__(self, config: "StateGainConfig | None" = None):
        self.config = config or StateGainConfig()

    def evaluate(self, obs: Observation, field: Field, t: int, kb: KnowledgeBase | None = None) -> PolicyDecision:
        cfg = self.config
        state_decision = decide_deposition(obs, field, t, cfg.selection_config)
        score = state_decision.state_gain

        return PolicyDecision(
            transmit=score > cfg.threshold,
            score=score,
            novelty=1.0 if state_decision.classification == selection.NOVEL else 0.0,
            classification=state_decision.classification,
        )


@dataclass
class M1Config:
    """Configuration for M1Policy: M1 = state_gain (field_sim.selection.
    decide_deposition's NOVEL/CORRECTION/REINFORCE/REFRESH/REDUNDANT
    classification) + reasoning_gain (DeltaU, via a nested
    ReasoningAwarePolicy) combined into one score:

        score = state_gain + DeltaU
        transmit iff score > threshold

    reasoning_config/selection_config configure the two sub-computations.
    ReasoningAwarePolicy contributes ONLY DeltaU (see its own docstring) -
    cold start for a first-time observation is handled entirely by
    state_gain's NOVEL case here, never by the nested policy.
    """

    reasoning_config: ReasoningAwareConfig = dc_field(default_factory=ReasoningAwareConfig)
    selection_config: SelectionConfig = dc_field(default_factory=SelectionConfig)
    threshold: float = 0.5


class M1Policy(CommunicationPolicy):
    """M1 - reasoning-aware communication: the restored pre-M1-era
    selective-deposition mechanism (field_sim.selection.decide_deposition's
    NOVEL/CORRECTION/REINFORCE/REFRESH/REDUNDANT classification ->
    state_gain), with reasoning-gain (via a nested ReasoningAwarePolicy)
    layered on top - M1 IS this combination.

    Unlike ReasoningAwarePolicy alone (reasoning_gain alone, which
    deadlocks whenever no single candidate has reasoning value in
    isolation - see ReasoningAwareConfig's docstring), state_gain alone
    already justifies transmitting a first-time (NOVEL) or
    contradicting (CORRECTION) observation regardless of its reasoning
    value, so cold start is not a structural failure mode here:
    reasoning_gain still matters (it can push a REINFORCE/REFRESH candidate
    that state_gain alone wouldn't justify over the threshold), it's just
    no longer the ONLY signal.
    """

    def __init__(self, kb: KnowledgeBase, reasoner: FieldReasoner | None = None, config: "M1Config | None" = None):
        self.kb = kb
        self.config = config or M1Config()
        self._reasoning = ReasoningAwarePolicy(kb, reasoner=reasoner, config=self.config.reasoning_config)

    def evaluate(self, obs: Observation, field: Field, t: int, kb: KnowledgeBase | None = None) -> PolicyDecision:
        cfg = self.config
        state_decision = decide_deposition(obs, field, t, cfg.selection_config)
        reasoning_decision = self._reasoning.evaluate(obs, field, t, kb)
        score = state_decision.state_gain + reasoning_decision.reasoning_gain

        return PolicyDecision(
            transmit=score > cfg.threshold,
            score=score,
            reasoning_gain=reasoning_decision.reasoning_gain,
            novelty=1.0 if state_decision.classification == selection.NOVEL else 0.0,
            classification=state_decision.classification,
        )
