"""Relation cardinality: which relations may hold several objects for the
same subject at once, and the storage key that follows from it.

Single-valued relations (locatedAt, movingToward, hasSpeed, ...) describe
ONE current value per subject - a new object replaces the old one, so their
storage key is the (subject, relation) pair, as it always was.

Multi-valued relations hold several objects at once:
  - near: field_sim.environment.Environment generates each subject's
    NEAREST entity, and the ontology declares near symmetric (one
    auto-generated reasoning rule), so the valid near triples are the
    symmetric closure of the nearest-neighbour relation - B can be near both
    A (because B is A's nearest) and C (B's own nearest).
  - potentialConflict: a Humanoid can be in potential conflict with every
    Pedestrian approaching its area at once.
  - canInteractWith: derived from near (R6), so it inherits near's
    multiplicity.
Their storage key is the full (subject, relation, object) triple, so two
different objects are two separate entries, never a conflict.

Kept in its own dependency-free module so field_sim.types (keys) and
field_sim.ontology (the schema) can both import it without a cycle.
"""

MULTI_VALUED_RELATIONS: frozenset[str] = frozenset({"near", "potentialConflict", "canInteractWith"})


def is_multi_valued(relation: str) -> bool:
    return relation in MULTI_VALUED_RELATIONS


def edge_key(subject: str, relation: str, obj: str) -> tuple:
    """Storage key for one trace: (subject, relation) for a single-valued
    relation, (subject, relation, object) for a multi-valued one."""
    if relation in MULTI_VALUED_RELATIONS:
        return (subject, relation, obj)
    return (subject, relation)
