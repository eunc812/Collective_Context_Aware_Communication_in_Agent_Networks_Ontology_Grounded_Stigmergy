"""M3: metadata-based context retrieval and a minimal agent decision - the
framework's collective-cognition-to-action use (the earlier communication-
context use, field_sim.communication.reactivate_suppressed(), has been
removed; this is the only remaining M3 mechanism).

    Dynamic Semantic Graph
            v
    Metadata-based Context Retrieval
            v
    Contextual Subgraph
            v
    Decision-level Inferred Edge Retrieval
            v
    Minimal Agent Decision

Deliberately minimal: a deterministic lookup over a small, fixed
decision-level property set, demonstrating only that a change in
collective cognition (M2, field_sim.reasoning / field_sim.field.Field.
reason()) can reach downstream agent behavior - not a planner, policy
network, or RL system.

No previous implementation of this action-context use survived in the
codebase to restore (only a leftover pytest-cache node id for a deleted
test, field_sim.ontology's old-domain retrieve_decision_context, using
"kind"/"radius" filtering that conflicts with this spec's area/time/
confidence criteria and the current VLA domain) - this module is written
fresh against the current environment/field/types, reusing field_sim.
types.FieldEdge and field_sim.field.Field.edges as-is rather than
introducing a parallel graph representation.

Retrieval is purely metadata-based, not ontology-typed: `area(e)` is a
point-in-rectangle lookup against edge.location (see field_sim.types.
resolve_area, shared with field_sim.environment) - no relation-range or
class-hierarchy inspection at all. This only became
possible once field_sim.environment.Environment started partitioning its
space into one rectangle per SpatialRegion instance and keeping every
entity's continuous location consistent with whichever area it is
locatedAt - before that, location carried no area information, and this
module had to fall back to inspecting each edge's ontology-typed
(subject, relation, object) triple instead.

area/time/confidence are reflected exactly once, in retrieve_context()
(Metadata-based Context Retrieval -> Contextual Subgraph above);
get_decision_edges() (Decision-level Inferred Edge Retrieval) does not
re-derive or bypass any of that - it only picks the querying agent's own
decision-level INFERRED edges out of the contextual subgraph it's handed,
so decision_edges is always a subset of retrieved_edges. A decision edge
whose own (possibly stale/inherited) location resolves outside the
querying agent's current context_area is filtered out at the context step
like anything else, same as any other edge - there is no subject-identity
exception.
"""

from dataclasses import dataclass

from field_sim.types import Bounds, FieldEdge, resolve_area

# Configurable retrieval thresholds - referenced everywhere below instead
# of repeating the numbers.
MAX_TIME_DIFFERENCE = 10
MIN_CONFIDENCE = 0.5

# Initial decision-level property set. No fixed relation-name priority
# any more - make_decision() below picks by confidence instead.
DECISION_MAP = {"shouldReroute": "REROUTE", "shouldSlowDown": "SLOW"}
DECISION_LEVEL_PROPERTIES = set(DECISION_MAP.keys())
DEFAULT_DECISION = "CONTINUE"


def retrieve_context(
    graph: dict[tuple[str, str], FieldEdge],
    agent_id: str,
    context_area: str,
    current_time: int,
    region_bounds: dict[str, Bounds],
    max_time_difference: int = MAX_TIME_DIFFERENCE,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[tuple[str, str], FieldEdge]:
    """G_context(i,t) = {e in G_t | area(e) = context_area AND
    current_time - time(e) < max_time_difference AND
    confidence(e) > min_confidence}.

    `agent_id` (the querying agent) is accepted per the spec's interface
    but deliberately NOT used to filter here - agent_id in a semantic
    trace's metadata is provenance (who originally observed it), not a
    retrieval criterion, so information contributed by OTHER agents stays
    retrievable (field_sim.types.FieldEdge.sources remains available on
    every returned edge for inspection). Only area + time + confidence
    gate retrieval.

    Returns a NEW dict (same FieldEdge objects, not copies) - a view into
    `graph`, never a mutation of it; nothing here writes to any FieldEdge
    or to `graph` itself, mirroring field_sim.reasoning.FieldReasoner's
    read-only discipline.
    """
    return {
        key: edge
        for key, edge in graph.items()
        if resolve_area(edge.location, region_bounds) == context_area
        and current_time - edge.last_observation_time < max_time_difference
        and edge.confidence > min_confidence
    }


def get_decision_edges(
    context_subgraph: dict[tuple[str, str], FieldEdge],
    agent_id: str,
) -> list[FieldEdge]:
    """Decision-level INFERRED edges about the querying agent - a pure
    relevance filter over `context_subgraph` (retrieve_context()'s already
    area/time/confidence-gated result), not a second, independently-gated
    query against the full graph. No bypass: a decision edge whose own
    location metadata (possibly stale/inherited through a premise chain -
    see field_sim.ontology.KnowledgeBase.infer_with_provenance()) resolves
    outside the querying agent's current context_area was already dropped
    by retrieve_context() and simply isn't here to find - area/time/
    confidence are reflected exactly once, at context-retrieval time.

    edge.fact_type == "inferred" (field_sim.types.FieldEdge, set by
    field_sim.field.Field.reason()/Field.set()) is the per-EDGE, runtime
    "was this particular edge actually produced by a rule firing"
    provenance marker - distinct from field_sim.ontology.INFERRED, which
    classifies a PROPERTY as inference-only by definition regardless of
    any single edge's history (see tests/test_ontology.py's
    test_reasoning_gain_counts_only_inferred_never_background). The
    spec's "edge_type == INFERRED" condition means the former.
    """
    return [
        edge
        for edge in context_subgraph.values()
        if edge.subject == agent_id
        and edge.relation in DECISION_LEVEL_PROPERTIES
        and edge.fact_type == "inferred"
    ]


def make_decision(decision_edges: list[FieldEdge]) -> str:
    """Pick the decision-level edge with the HIGHEST confidence and map its
    relation through DECISION_MAP - not a planner, just a confidence-based
    arbitration. Confidence is the one signal this whole framework already
    uses everywhere else to weigh competing information (state_gain's
    margins, MIN_CONFIDENCE filtering here in this same module), so
    reusing it to arbitrate between competing decision-level edges is
    consistent with that rather than an unrelated, unexplained fixed
    relation-name order (a barely-qualifying shouldReroute no longer
    automatically beats a highly-confident shouldSlowDown).

    Ties (equal confidence) fall back to list order (max() returns the
    first maximal element) - not itself meaningful, just deterministic.
    No decision edge -> CONTINUE.
    """
    if not decision_edges:
        return DEFAULT_DECISION
    best = max(decision_edges, key=lambda edge: edge.confidence)
    return DECISION_MAP[best.relation]


@dataclass
class M3Result:
    """What field_sim.simulation should later be able to inspect: what
    collective information the agent retrieved, which inferred edge(s)
    triggered the decision, and what final action was selected."""

    agent_id: str
    context_area: str
    retrieved_edges: list[FieldEdge]
    decision_edges: list[FieldEdge]
    decision: str


def m3_decide(
    graph: dict[tuple[str, str], FieldEdge],
    agent_id: str,
    context_area: str,
    current_time: int,
    region_bounds: dict[str, Bounds],
    max_time_difference: int = MAX_TIME_DIFFERENCE,
    min_confidence: float = MIN_CONFIDENCE,
) -> M3Result:
    """The full M3 action-context pipeline: retrieve_context ->
    get_decision_edges -> make_decision, bundled into one result.

    get_decision_edges() now runs ON `context_subgraph`, not the full
    `graph` - decision_edges is therefore always a subset of retrieved_edges
    by construction (no separate union needed), and area/time/confidence
    are applied exactly once, in retrieve_context().
    """
    context_subgraph = retrieve_context(
        graph, agent_id, context_area, current_time, region_bounds, max_time_difference, min_confidence
    )
    decisions = get_decision_edges(context_subgraph, agent_id)
    return M3Result(
        agent_id=agent_id,
        context_area=context_area,
        retrieved_edges=list(context_subgraph.values()),
        decision_edges=decisions,
        decision=make_decision(decisions),
    )
