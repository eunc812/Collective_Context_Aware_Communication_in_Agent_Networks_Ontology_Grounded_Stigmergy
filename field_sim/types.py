"""Shared data structures: facts, observations, field edges."""

from dataclasses import dataclass, field

from field_sim.cardinality import edge_key


@dataclass(frozen=True)
class Location:
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


# (x_min, x_max, y_min, y_max) - a region's axis-aligned rectangle. Shared
# by field_sim.environment (which computes these) and field_sim.decision
# (which resolves a Location against them) so neither has to duplicate the
# other's declaration.
Bounds = tuple[float, float, float, float]


def resolve_area(location: Location, region_bounds: dict[str, Bounds]) -> str | None:
    """Which region's rectangle contains `location` - a plain
    point-in-rectangle lookup, nothing else. Shared by field_sim.environment
    (Environment._drift_subject, resolving a drifted position back to its
    area) and field_sim.decision (retrieve_context/receive_field_context,
    resolving a FieldEdge's or a querying agent's location) - previously
    duplicated as two separate implementations (one on each side) to avoid
    either module importing the other; living here instead, in the shared
    types module both already depend on, removes the duplication without
    introducing that dependency.

    `region_bounds` is normally field_sim.environment.Environment.
    region_bounds: one non-overlapping rectangle per SpatialRegion
    instance, gaplessly tiling the whole space for any nonzero region count
    (field_sim.environment._partition_regions()). Adjacent rectangles share
    a boundary coordinate exactly (closed intervals on both sides); a point
    exactly on that line matches whichever region is checked first
    (insertion order) - negligible in practice since locations are sampled
    from the open interior, never generated exactly on a boundary.

    Returns None if `location` falls inside no known rectangle - only
    possible for a location genuinely outside the tiled space, or an empty
    `region_bounds`.
    """
    for area, (x_min, x_max, y_min, y_max) in region_bounds.items():
        if x_min <= location.x <= x_max and y_min <= location.y <= y_max:
            return area
    return None


@dataclass
class Fact:
    """A single ground-truth (subject, relation, object) triple in the environment."""

    subject: str
    relation: str
    object: str
    location: Location
    last_changed_time: int

    @property
    def key(self) -> tuple[str, str]:
        """Facts are identified by (subject, relation); object is the current
        value. Deliberately NOT field_sim.cardinality.edge_key(): the
        environment generates exactly one value per (subject, relation) for
        every observable relation (near = the single nearest entity), and
        mutates fact.object in place when it changes - a triple key would go
        stale the moment the value moved. The multi-valued view of the world
        (e.g. near's symmetric closure) is built from these facts by
        field_sim.evaluation.reference_state()."""
        return (self.subject, self.relation)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class RawObservation:
    """A raw, agent-local measurement in that agent's own modality/vocabulary.

    Different agents may use different sensors or representations for the
    same underlying fact (e.g. a numeric code vs. an abbreviation vs. plain
    text) - this is the heterogeneous form, before it has been grounded in
    the shared knowledge base.
    """

    subject: str
    modality: str
    raw_relation: str
    raw_value: str
    location: Location
    obs_time: int
    observation_quality: float  # sensor-side confidence, pre-mapping
    agent_id: str


@dataclass
class Observation:
    """A candidate semantic trace: a RawObservation mapped onto the shared
    vocabulary (subject, relation, object), ready to be compared against the
    field.

    Per the design memo's (s, r, o, m) semantic trace - metadata is (area,
    time, sensor observation confidence, agent id), a SINGLE confidence
    number, not two separate components. Where the raw label mapping
    (field_sim.ontology.value_aliases) is uncertain, that uncertainty
    is folded into this single number at grounding time (KnowledgeBase.
    map_to_candidate()) rather than carried as its own field.
    """

    subject: str
    relation: str
    object: str
    location: Location
    obs_time: int
    confidence: float
    agent_id: str

    @property
    def key(self) -> tuple:
        """Storage key (field_sim.cardinality.edge_key): the (subject,
        relation) pair for a single-valued relation, the full triple for a
        multi-valued one."""
        return edge_key(self.subject, self.relation, self.object)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.subject, self.relation)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class FieldEdge:
    """An aggregated triple living in the shared field (the collective
    cognition state) - subject/relation/object plus confidence metadata,
    per the collective-cognition-aware communication framework's "local
    cognition is the subject-relation-object triple, having metadata of
    confidence at the edge."

    Written by field_sim.field.Field.set() - a single, unconditional
    upsert on this edge's storage key (field_sim.cardinality.edge_key):
    last-write-wins per (subject, relation) for a single-valued relation,
    one independent edge per triple for a multi-valued one. Whether a write
    happens at all is decided
    upstream (field_sim.selection.decide_deposition, field_sim.
    communication.ReasoningAwarePolicy/M1Policy) - Field.set() itself does
    no classification, and there is still no competing-claims/conflict-group
    tracking anywhere in this codebase. `support_count`/`sources` are plain
    provenance metadata (how many writes, which agents) - descriptive, not
    used to decide how a write is applied.
    """

    subject: str
    relation: str
    object: str
    location: Location
    confidence: float
    last_observation_time: int
    last_deposition_time: int
    # when the CURRENT `object` value was first written - unlike
    # last_deposition_time, a write that repeats the SAME value (REINFORCE/
    # REFRESH, field_sim.selection) does NOT advance this. Lets
    # decide_deposition's staleness-based correction bypass measure "how
    # long has THIS belief gone unchallenged" separately from "how recently
    # was anything written here at all" - a persistently-repeated wrong
    # value can keep last_deposition_time fresh forever (REFRESH) without
    # ever resetting value_since, so a competing correction's staleness
    # clock still advances (see field_sim.selection.decide_deposition).
    value_since: int
    support_count: int = 0
    sources: set[str] = field(default_factory=set)
    # M2 (field_sim.reasoning): "observed" for anything an agent
    # transmitted, "inferred" for anything a rule derived (Field.reason() /
    # FieldReasoner). A later write to the SAME value never demotes this
    # from "inferred" back to "observed" - once collective reasoning has
    # produced an inference, a fresh observation merely reconfirming the
    # same value isn't a retraction of the inference having occurred, so
    # it shouldn't erase that provenance either (field_sim.field.Field.
    # set() enforces this directly now - see its own docstring for why:
    # left unenforced, a same-value reconfirmation silently overwriting
    # this to "observed" was read by field_sim.communication.
    # ReasoningAwarePolicy's counterfactual as a loss of reasoning value,
    # producing a reasoning_gain that should have been exactly 0 but came
    # out negative instead). A genuine VALUE change (a correction) still
    # overwrites fact_type from whatever is being written, same as always.
    fact_type: str = "observed"
    # M2 provenance, INFERRED edges only (both stay at their defaults -
    # None / empty set - for an observed/background edge, which has no
    # rule or premises to record): which field_sim.ontology.ReasoningRule
    # produced this edge, and the (subject, relation) keys of the premise
    # FieldEdges its firing actually used. Filled in by field_sim.field.
    # Field.reason(), the only place that sees both the rule and its
    # premises at once (KnowledgeBase.infer_with_provenance() knows this
    # per firing, but the return value Field.set() receives is just the
    # already-built candidate Observation - see reason()'s docstring).
    # Unlike `sources` (agent-level, unioned across every firing an edge
    # has ever had), rule_id/condition_sources reflect only the MOST
    # RECENT firing that wrote this edge - if a different rule later
    # re-derives the same (subject, relation), it replaces both.
    rule_id: str | None = None
    condition_sources: set[tuple] = field(default_factory=set)

    @property
    def key(self) -> tuple:
        """Storage key in Field.edges - see Observation.key."""
        return edge_key(self.subject, self.relation, self.object)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.subject, self.relation)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)
