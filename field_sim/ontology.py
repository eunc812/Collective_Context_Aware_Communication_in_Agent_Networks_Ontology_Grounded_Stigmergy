"""The VLA humanoid domain's ontology, in full: Class/Instance layer,
Object/Datatype properties, reasoning rules, and the KnowledgeBase that
grounds heterogeneous raw observations into it - plus the specification
layer on top (property characteristics, relationships, alias/grounding
table, observed/inferred/background classification, and validation
helpers). One file, one set of concepts: this used to be split across
field_sim.knowledge_base (the live grounding/reasoning mechanism) and
field_sim.ontology (a later, additive specification layer with its own,
disconnected alias table) - merged here so there's a single source of
truth, most concretely: KnowledgeBase.relation_aliases below is now
DERIVED from property_alias_table instead of duplicating it.

Holds:
  - class_hierarchy:      the Class layer - subclass relationships (Entity ->
                          Agent -> Human -> Pedestrian, etc, see
                          ClassHierarchy.is_a()), on KnowledgeBase.
  - subject_class:        the Instance layer - subject -> its class
                          (Pedestrian / Package / Room / ...), on KnowledgeBase.
  - object_properties:    ObjectProperty specs (entity -> entity) - domain/
                          range are CLASSES, checked subclass-aware via
                          class_hierarchy (a "carrying" scoped to Humanoid
                          also accepts a Humanoid subclass, if one existed).
  - datatype_properties:  DatatypeProperty specs (entity -> literal value) -
                          domain is a class, range is a primitive Python
                          type (e.g. float).
  - property_characteristics / property_relationships / property_alias_table /
    property_source_type: the declarative specification layer (module-level
                          dicts, see each section below).
  - class_disjoint_with:  Class-layer disjointness (e.g. Humanoid/Human,
                          MovableObject/StaticObject/Obstacle) - a SCHEMA
                          (TBox) declaration checked against class_hierarchy
                          itself (check_class_disjointness()), not against
                          live instance data: KnowledgeBase.subject_class
                          holds exactly one class per subject and
                          class_hierarchy is single-inheritance, so no
                          instance could ever structurally satisfy is_a()
                          for two different, non-ancestor-related classes -
                          a live per-instance check would be exactly as
                          vacuous as the removed check_functional_property
                          (see D2). Property-level disjointness
                          (movingToward/movingAwayFrom, property_relationships'
                          "disjointWith") is the live/ABox counterpart -
                          checked against actual FieldEdges instead, via
                          check_disjoint_properties(), since nothing stops
                          two DIFFERENT edges existing for the same subject.
  - relation_aliases:     modality -> raw relation label -> canonical
                          property, on KnowledgeBase - derived from
                          property_alias_table (single "canonical" modality;
                          no per-modality label variants exist in this
                          domain, unlike the older status/intensity domain).
  - reasoning_rules:      M2's sole reasoning mechanism, on KnowledgeBase -
                          variable-join rules over arbitrary antecedent
                          conjunctions, with an optional type_constraints
                          filter (see ReasoningRule). Includes one
                          auto-generated symmetric rule per
                          ObjectPropertySpec(symmetric=True) (e.g. "near"),
                          plus the reasoning rule spec's R0-R6.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from field_sim.cardinality import edge_key
from field_sim.types import FieldEdge, Observation, RawObservation

# --- source-type constants ---------------------------------------------------

OBSERVED = "OBSERVED"
INFERRED = "INFERRED"
UNKNOWN_PROPERTY = "UNKNOWN_PROPERTY"


# ============================================================
# CORE ONTOLOGY DATACLASSES (Class/Instance/Property/Reasoning)
# ============================================================

@dataclass
class ObjectPropertySpec:
    """domain/range for an ObjectProperty (entity -> entity). `range` is a
    CLASS the object must belong to - checked via ClassHierarchy.is_a
    against the OBJECT's own registered instance class (KnowledgeBase.
    subject_class), the same way `domain` is checked against the subject's
    class. An unregistered object (never passed to register_subject()) is
    accepted rather than rejected - there is nothing to check it against,
    same policy as an unregistered subject.
    `symmetric=True` auto-generates a ReasoningRule so (a, relation, b)
    always entails (b, relation, a) via M2 (see KnowledgeBase.__init__).
    `inferred=True` marks a property that only ever comes from a reasoning
    rule (ReasoningRule), never a direct observation - it gets no
    relation_aliases entry (nothing can ground a raw reading into it) and
    field_sim.environment.Environment never generates ground truth for it
    directly."""
    domain: str
    range: str
    symmetric: bool = False
    inferred: bool = False


@dataclass
class DatatypePropertySpec:
    """domain/range for a DatatypeProperty (entity -> literal value).
    `range` is a primitive Python type (e.g. `float`), not a class - the
    raw value must parse as that type to be accepted."""
    domain: str
    range: type


@dataclass
class ClassHierarchy:
    """subclass relationships: parent[cls] = its direct superclass, or
    absent (parent.get(cls) is None) for a root class. A class with no
    entry here is its own single-node hierarchy.

    This is the Class layer of Class/Instance - KnowledgeBase.subject_class
    (instance -> its class) is the Instance layer; is_a() is what makes a
    domain/range constraint scoped to a SUPERCLASS (e.g. "Agent") accept an
    instance of any of its subclasses (e.g. "Pedestrian") instead of
    requiring an exact class-name match.
    """

    parent: dict[str, str] = dc_field(default_factory=dict)

    def ancestors(self, cls: str) -> set[str]:
        """`cls` itself, plus every superclass up to the root."""
        result = {cls}
        current = self.parent.get(cls)
        while current is not None and current not in result:
            result.add(current)
            current = self.parent.get(current)
        return result

    def is_a(self, cls: str, target: str) -> bool:
        """Does `cls` equal or subclass `target`?"""
        return target in self.ancestors(cls)


# class_disjoint_with is deliberately independent of ClassHierarchy.parent
# (an explicit, hand-authored declaration - not derived from `parent`) so
# check_class_disjointness() below is checking one source of truth against
# another, not a tree property against itself. Every pair of classes that
# intuitively can't describe the same instance at once - a physical object
# can't also be a spatial region it sits in, a MovableObject can't also be
# a StaticObject, etc. Declared both directions for convenient lookup.
class_disjoint_with: dict[str, list[str]] = {
    "Agent": ["PhysicalObject", "SpatialRegion"],
    "PhysicalObject": ["Agent", "SpatialRegion"],
    "SpatialRegion": ["Agent", "PhysicalObject"],
    "Humanoid": ["Human"],
    "Human": ["Humanoid"],
    "MovableObject": ["StaticObject", "Obstacle"],
    "StaticObject": ["MovableObject", "Obstacle"],
    "Obstacle": ["MovableObject", "StaticObject"],
}


def get_class_disjoint_with(cls: str) -> list[str]:
    """Empty list (not an error) for a class with no declared disjointness."""
    return class_disjoint_with.get(cls, [])


@dataclass(frozen=True)
class Atom:
    """One triple pattern used by a ReasoningRule (M2's generalized join,
    see field_sim.reasoning.FieldReasoner and KnowledgeBase.infer() below).

    subject/object are either a variable - a token starting with "?", e.g.
    "?x" - or a literal constant (e.g. "Task1"). relation is always a
    literal.
    """

    relation: str
    subject: str
    object: str


@dataclass
class ReasoningRule:
    """A variable-based conjunctive forward-chaining rule, and this domain's
    SOLE reasoning mechanism: every antecedent atom is matched against the
    field's current triples with unification, and each satisfying variable
    binding instantiates the consequent as a new candidate trace - arbitrary
    antecedent conjunctions, non-spatial joins, cross-joins with no shared
    variable, and chains where one rule's consequent feeds another rule's
    antecedent.

    type_constraints (variable -> required class) is an ADDITIONAL filter on
    top of triple-pattern unification: a binding only satisfies the rule if
    every constrained variable's bound value is a REGISTERED instance whose
    class is-a the required class (subclass-aware, via class_hierarchy).
    This matters because a triple pattern like (?x, movingToward, ?a) only
    checks that a movingToward edge exists, not that ?a is specifically an
    Area rather than some other SpatialRegion (Room, Corridor, ...) -
    domain/range on the underlying ObjectProperty is necessarily broader
    (SpatialRegion) than what a SPECIFIC rule needs (Area, for R1). An
    unconstrained variable
    passes trivially; a CONSTRAINED variable bound to an unregistered
    subject fails closed (rejected) - unlike map_to_candidate()'s
    benefit-of-the-doubt policy for raw observations, a reasoning rule
    creates a new derived fact treated as ground truth from then on, so an
    unverifiable type doesn't get to satisfy the constraint by default.
    """

    name: str
    antecedents: tuple[Atom, ...]
    consequent: Atom
    type_constraints: dict = dc_field(default_factory=dict)


def _is_variable(token: str) -> bool:
    return token.startswith("?")


def _solve_rule(
    antecedents: list[Atom],
    bindings: dict[str, str],
    premises: list[FieldEdge],
    facts_by_relation: dict[str, list[FieldEdge]],
):
    """Naive backtracking join: yields (bindings, premises) for every way to
    satisfy `antecedents` against `facts_by_relation`, one edge per
    antecedent. Small and unindexed on purpose - toy-domain fact counts are
    tiny, so a linear scan per antecedent is plenty and keeps this readable."""
    if not antecedents:
        yield dict(bindings), list(premises)
        return

    atom, rest = antecedents[0], antecedents[1:]
    for edge in facts_by_relation.get(atom.relation, []):
        if any(edge is used for used in premises):
            continue  # a rule can't use the same fact as two of its own antecedents
        new_bindings = dict(bindings)
        ok = True
        for pattern, value in ((atom.subject, edge.subject), (atom.object, edge.object)):
            if _is_variable(pattern):
                if pattern in new_bindings and new_bindings[pattern] != value:
                    ok = False
                    break
                new_bindings[pattern] = value
            elif pattern != value:
                ok = False
                break
        if not ok:
            continue
        yield from _solve_rule(rest, new_bindings, premises + [edge], facts_by_relation)


# ============================================================
# 1. PROPERTY CHARACTERISTICS
# ============================================================
# Only what's explicitly warranted - no characteristic is added for a
# property just because it COULD have one. Cardinality (single- vs
# multi-valued) is NOT declared here but in field_sim.cardinality.
# MULTI_VALUED_RELATIONS, because it is what the storage key of every trace
# is built from (field_sim.cardinality.edge_key, used by field_sim.types)
# and has to be importable without this module. Every relation not listed
# there is single-valued: Field.edges keeps one object per (subject,
# relation) for it, and a new object replaces the old one.

property_characteristics: dict[str, dict[str, bool]] = {
    "near": {"symmetric": True, "irreflexive": True},
    "movingToward": {"irreflexive": True},
    "movingAwayFrom": {"irreflexive": True},
    # locatedAt, hasSpeed, blocks, carrying, interactingWith: no special
    # characteristic.
    # approachingArea, pathBlocked, shouldSlowDown, shouldReroute,
    # canInteractWith: no special characteristic.
    # potentialConflict: left directional (no symmetric=True) - only
    # meaningful as mutual if the implementation chooses to interpret it
    # that way, which this one does not.
}


def get_property_characteristics(property_name: str) -> dict[str, bool]:
    """Empty dict (not an error) for a property with no declared
    characteristics - "no special characteristic" IS a valid answer."""
    return property_characteristics.get(property_name, {})


# ============================================================
# 2. PROPERTY RELATIONSHIPS
# ============================================================
# Kept separate from characteristics (those describe ONE property's own
# shape; these describe relationships BETWEEN two properties). No new
# super-properties (SpatialRelation, MotionRelation, ...) - subPropertyOf
# is supported as a mechanism but nothing uses it yet.

property_relationships: dict[str, dict[str, list[str] | str]] = {
    # the same entity can't be heading toward AND away from the same
    # target at once - see check_disjoint_properties() below.
    "movingToward": {"disjointWith": ["movingAwayFrom"]},
    "movingAwayFrom": {"disjointWith": ["movingToward"]},
}


def get_property_relationships(property_name: str) -> dict[str, list[str] | str]:
    """Empty dict (not an error) for a property with no declared relationships."""
    return property_relationships.get(property_name, {})


# ============================================================
# 3. PROPERTY ALIAS / GROUNDING TABLE
# ============================================================
# The SINGLE source of truth for raw-label -> canonical-property grounding
# in this domain - KnowledgeBase.relation_aliases (below) is derived from
# this table rather than duplicating it, so canonicalize_property() and the
# live map_to_candidate() pipeline can never drift apart.
#
# canonical property -> { modality -> [raw labels a sensor of that modality
# would use] }. Three modalities, matching field_sim.agent.MODALITIES:
#   - "canonical": plain-language labels/synonyms (several aliases each)
#   - "numeric":   a fixed 3-digit code per property (a cheap sensor that
#                  only knows codes, not words)
#   - "abbrev":    a short abbreviation per property (a terse log format)
# Every alias is unique across the WHOLE table (all properties, all
# modalities) - canonicalize_property() below stays modality-agnostic on
# purpose, since a label alone is already enough to identify the property
# no matter which sensor produced it.

property_alias_table: dict[str, dict[str, list[str]]] = {
    "locatedAt": {
        "canonical": ["locatedAt", "at", "in", "inside", "located_in"],
        "numeric": ["101"],
        "abbrev": ["loc"],
    },
    "near": {
        "canonical": ["near", "nearby", "close_to", "close"],
        "numeric": ["103"],
        "abbrev": ["nr"],
    },
    "movingToward": {
        "canonical": ["movingToward", "approaching", "heading_to", "moving_to", "going_toward"],
        "numeric": ["104"],
        "abbrev": ["mvt"],
    },
    "movingAwayFrom": {
        "canonical": ["movingAwayFrom", "leaving", "moving_away", "heading_away"],
        "numeric": ["105"],
        "abbrev": ["mvf"],
    },
    "blocks": {
        "canonical": ["blocks", "blocking", "obstructs", "obstructing", "occupies_path"],
        "numeric": ["106"],
        "abbrev": ["blk"],
    },
    "carrying": {
        "canonical": ["carrying", "holding", "transporting"],
        "numeric": ["107"],
        "abbrev": ["cry"],
    },
    "interactingWith": {
        "canonical": ["interactingWith", "interacting_with", "manipulating"],
        "numeric": ["108"],
        "abbrev": ["ix"],
    },
    "hasSpeed": {
        "canonical": ["hasSpeed", "speed", "moving_speed"],
        "numeric": ["109"],
        "abbrev": ["spd"],
    },
}

# raw alias -> canonical property, flattened across every modality - built
# once from property_alias_table.
_alias_to_canonical: dict[str, str] = {
    alias: canonical
    for canonical, by_modality in property_alias_table.items()
    for aliases in by_modality.values()
    for alias in aliases
}


def canonicalize_property(raw_relation: str) -> str:
    """Deterministic raw label -> canonical property lookup. Returns
    UNKNOWN_PROPERTY (never guesses) if `raw_relation` isn't in the
    grounding table at all - the caller (an ontology validation stage)
    decides whether to reject or log the triple.

    >>> canonicalize_property("heading_to")
    'movingToward'
    >>> canonicalize_property("inside")
    'locatedAt'
    """
    return _alias_to_canonical.get(raw_relation, UNKNOWN_PROPERTY)


# ============================================================
# 4. OBSERVED VS INFERRED CLASSIFICATION
# ============================================================

property_source_type: dict[str, str] = {
    "locatedAt": OBSERVED,
    "near": OBSERVED,
    "movingToward": OBSERVED,
    "movingAwayFrom": OBSERVED,
    "blocks": OBSERVED,
    "carrying": OBSERVED,
    "interactingWith": OBSERVED,
    "hasSpeed": OBSERVED,
    "approachingArea": INFERRED,
    "pathBlocked": INFERRED,
    "shouldReroute": INFERRED,
    "potentialConflict": INFERRED,
    "shouldSlowDown": INFERRED,
    "canInteractWith": INFERRED,
}


def get_property_source_type(property_name: str) -> str:
    """UNKNOWN_PROPERTY (not a crash) for a property this table doesn't
    know about - callers should treat that the same as "not classified"."""
    return property_source_type.get(property_name, UNKNOWN_PROPERTY)


def is_observed_property(property_name: str) -> bool:
    return get_property_source_type(property_name) == OBSERVED


def is_inferred_property(property_name: str) -> bool:
    return get_property_source_type(property_name) == INFERRED


# ============================================================
# ENTITY GROUNDING (raw entity label -> canonical instance id)
# ============================================================
# Empty by default: this domain's raw observations already use canonical
# instance ids (Humanoid_1, Ped_1, ...) directly, so nothing needs
# aliasing today. The mechanism is real, not decorative - register an
# alias (e.g. a tracker's own object id) and map_to_candidate() below will
# resolve it to the canonical instance before anything else runs. An
# unaliased raw subject passes through unchanged (same benefit-of-the-doubt
# policy as an unregistered class).

entity_aliases: dict[str, str] = {}


def canonicalize_entity(raw_subject: str) -> str:
    """raw entity label -> canonical instance id. Unlike
    canonicalize_property(), an unknown label is NOT rejected - it's
    returned unchanged, since a not-yet-aliased entity id is still usable
    (it just won't have a registered class until KnowledgeBase.
    register_subject() is called for it, same as any other unregistered
    subject)."""
    return entity_aliases.get(raw_subject, raw_subject)


# ============================================================
# KNOWLEDGE BASE: grounds raw observations into this ontology, and
# evaluates M2 reasoning rules against the field
# ============================================================

class KnowledgeBase:
    def __init__(self):
        # Class layer: subclass relationships (the VLA humanoid ontology).
        self.class_hierarchy = ClassHierarchy(parent={
            # PhysicalEntity: the two branches of Entity that can meaningfully
            # occupy a SpatialRegion (be "locatedAt" one) - added so
            # locatedAt's domain can be scoped to exactly Agent+PhysicalObject
            # instead of the too-broad "Entity" (which also covers
            # SpatialRegion itself, producing vacuous self-referential facts
            # like "Area_1 locatedAt Area_2" - see ObjectPropertySpec's
            # locatedAt entry below).
            "PhysicalEntity": "Entity",
            "Agent": "PhysicalEntity",
            "Humanoid": "Agent",
            "Human": "Agent",
            "Pedestrian": "Human",
            "PhysicalObject": "PhysicalEntity",
            "MovableObject": "PhysicalObject",
            "Package": "MovableObject",
            "Container": "MovableObject",
            "Tool": "MovableObject",
            "StaticObject": "PhysicalObject",
            "Furniture": "StaticObject",
            "Wall": "StaticObject",
            "Door": "StaticObject",
            "Obstacle": "PhysicalObject",
            "SpatialRegion": "Entity",
            "Area": "SpatialRegion",
            "Room": "SpatialRegion",
            "Corridor": "SpatialRegion",
            "WorkZone": "SpatialRegion",
            "Passage": "SpatialRegion",
        })
        # Instance layer: subject -> its class (Pedestrian / Package / Room /
        # ... - any class_hierarchy key). register_subject() populates this
        # generically; nothing is pre-registered here.
        self.subject_class: dict[str, str] = {}

        # ObjectProperty specs: entity -> entity, range checked against the
        # OBJECT's own registered class (subclass-aware, see
        # ObjectPropertySpec). symmetric=True auto-generates a ReasoningRule
        # below.
        self.object_properties: dict[str, ObjectPropertySpec] = {
            # domain=PhysicalEntity (Agent+PhysicalObject), NOT the broader
            # "Entity" - a SpatialRegion instance has no meaningful locatedAt
            # of its own (see PhysicalEntity's docstring above), so it simply
            # gets no locatedAt fact at all now, instead of a vacuous
            # self-referential one like "Area_1 locatedAt Area_2".
            "locatedAt": ObjectPropertySpec(domain="PhysicalEntity", range="SpatialRegion"),
            "near": ObjectPropertySpec(domain="Entity", range="Entity", symmetric=True),
            "movingToward": ObjectPropertySpec(domain="Agent", range="SpatialRegion"),
            "movingAwayFrom": ObjectPropertySpec(domain="Agent", range="SpatialRegion"),
            "blocks": ObjectPropertySpec(domain="PhysicalObject", range="SpatialRegion"),
            "carrying": ObjectPropertySpec(domain="Humanoid", range="MovableObject"),
            "interactingWith": ObjectPropertySpec(domain="Humanoid", range="PhysicalObject"),
            # inferred-only properties (reasoning rule spec section 1) -
            # never directly observed, only ever produced by the reasoning
            # rules below (R1-R6).
            "approachingArea": ObjectPropertySpec(domain="Agent", range="Area", inferred=True),
            "pathBlocked": ObjectPropertySpec(domain="Humanoid", range="SpatialRegion", inferred=True),
            "potentialConflict": ObjectPropertySpec(domain="Agent", range="Agent", inferred=True),
            "shouldSlowDown": ObjectPropertySpec(domain="Humanoid", range="SpatialRegion", inferred=True),
            "shouldReroute": ObjectPropertySpec(domain="Humanoid", range="SpatialRegion", inferred=True),
            "canInteractWith": ObjectPropertySpec(domain="Humanoid", range="PhysicalObject", inferred=True),
        }
        # DatatypeProperty specs: entity -> literal value (range is a
        # primitive Python type, not a class).
        self.datatype_properties: dict[str, DatatypePropertySpec] = {
            "hasSpeed": DatatypePropertySpec(domain="Agent", range=float),
        }

        # modality -> raw_relation_label -> canonical_relation. Derived
        # from property_alias_table (module-level, above) - single source
        # of truth, so this and canonicalize_property() can't drift apart.
        # Three modalities ("canonical"/"numeric"/"abbrev", matching
        # field_sim.agent.MODALITIES) each get their own raw-label -> canonical
        # sub-table, built by flattening property_alias_table's per-property,
        # per-modality alias lists.
        self.relation_aliases: dict[str, dict[str, str]] = {}
        for canonical, by_modality in property_alias_table.items():
            for modality, aliases in by_modality.items():
                table = self.relation_aliases.setdefault(modality, {})
                for alias in aliases:
                    table[alias] = canonical

        # No raw-value aliasing needed for this domain: an ObjectProperty's
        # value is an instance id used directly (canonical already), and a
        # DatatypeProperty's value is a raw typed literal (parsed directly,
        # see DatatypePropertySpec.range) - neither goes through a
        # label -> canonical-value lookup the way the old categorical
        # status/intensity domain's abbreviations did. Kept as an empty
        # structure (not removed) - encode() below still consults it
        # generically for any modality/relation that DID declare one.
        self.value_aliases: dict[str, dict[str, dict[str, tuple[str, float]]]] = {}

        # Reverse tables used only to synthesize raw sensor readings for the
        # simulation - a real deployment wouldn't need this, agents would
        # natively emit raw labels from their own sensors.
        self._encode_relation: dict[str, dict[str, str]] = {}
        for modality, aliases in self.relation_aliases.items():
            self._encode_relation[modality] = {v: k for k, v in aliases.items()}

        self._encode_value: dict[str, dict[str, dict[str, str]]] = {}
        for modality, by_relation in self.value_aliases.items():
            self._encode_value[modality] = {}
            for canonical_relation, values in by_relation.items():
                self._encode_value[modality][canonical_relation] = {
                    canonical_object: raw_value for raw_value, (canonical_object, _) in values.items()
                }

        # M2's generalized rules: one auto-generated per symmetric
        # ObjectProperty (R0_near_symmetric), plus the reasoning rule spec's
        # R1-R6 (R0/partOf_transitive was removed along with `partOf` itself
        # - see R1's comment below; R7, the numeric-comparison speed-aware
        # rule, is intentionally omitted per the spec's own "can be omitted
        # from the first implementation").
        self.reasoning_rules: list[ReasoningRule] = [
            ReasoningRule(f"{relation}_symmetric", (Atom(relation, "?a", "?b"),), Atom(relation, "?b", "?a"))
            for relation, spec in self.object_properties.items()
            if spec.symmetric
        ] + [
            # R1: movingToward(x,a) -> approachingArea(x,a), a: Area.
            # Single-hop since `partOf` (the old 2-hop movingToward(x,r) &
            # partOf(r,a) chain) was removed - all regions are same-size
            # grid cells with no real geometric containment, so there was
            # nothing honest left to derive `partOf` from. movingToward
            # itself is now geometrically grounded (Environment picks
            # whichever region the subject's real velocity vector points
            # at most directly - see Environment._moving_toward_region),
            # so a direct type-constrained match on "is that region an
            # Area" reuses the SAME type_constraints filtering mechanism
            # R4/R5 already use, with no new engine code.
            ReasoningRule("approaching_area",
                (Atom("movingToward", "?x", "?a"),),
                Atom("approachingArea", "?x", "?a"),
                type_constraints={"x": "Agent", "a": "Area"}),
            # R2: movingToward(h,r) & blocks(o,r) -> pathBlocked(h,r)
            ReasoningRule("path_blocked",
                (Atom("movingToward", "?h", "?r"), Atom("blocks", "?o", "?r")),
                Atom("pathBlocked", "?h", "?r"),
                type_constraints={"h": "Humanoid", "o": "PhysicalObject", "r": "SpatialRegion"}),
            # R3: pathBlocked(h,r) -> shouldReroute(h,r)
            ReasoningRule("should_reroute",
                (Atom("pathBlocked", "?h", "?r"),),
                Atom("shouldReroute", "?h", "?r"),
                type_constraints={"h": "Humanoid", "r": "SpatialRegion"}),
            # R4: locatedAt(h,a) & approachingArea(p,a) -> potentialConflict(h,p)
            ReasoningRule("potential_conflict",
                (Atom("locatedAt", "?h", "?a"), Atom("approachingArea", "?p", "?a")),
                Atom("potentialConflict", "?h", "?p"),
                type_constraints={"h": "Humanoid", "p": "Pedestrian", "a": "Area"}),
            # R5: potentialConflict(h,p) & locatedAt(h,a) -> shouldSlowDown(h,a)
            ReasoningRule("should_slow_down",
                (Atom("potentialConflict", "?h", "?p"), Atom("locatedAt", "?h", "?a")),
                Atom("shouldSlowDown", "?h", "?a"),
                type_constraints={"h": "Humanoid", "p": "Pedestrian", "a": "SpatialRegion"}),
            # R6: near(h,o) -> canInteractWith(h,o)
            ReasoningRule("can_interact",
                (Atom("near", "?h", "?o"),),
                Atom("canInteractWith", "?h", "?o"),
                type_constraints={"h": "Humanoid", "o": "PhysicalObject"}),
        ]

    def register_subject(self, subject: str, subject_class: str) -> None:
        self.subject_class[subject] = subject_class

    def encode(self, modality: str, canonical_relation: str, canonical_object: str) -> tuple[str, str] | None:
        """Simulation-only helper: what would a sensor of `modality` report for this fact?"""
        raw_relation = self._encode_relation.get(modality, {}).get(canonical_relation)
        if raw_relation is None:
            return None
        value_table = self._encode_value.get(modality, {}).get(canonical_relation)
        if value_table is None:
            # no raw-value aliasing defined for this relation (this domain's
            # ObjectProperty/DatatypeProperty values ARE the canonical form
            # already, see value_aliases' docstring in __init__) - the
            # object passes through unchanged rather than failing to encode.
            return raw_relation, canonical_object
        raw_value = value_table.get(canonical_object)
        if raw_value is None:
            return None
        return raw_relation, raw_value

    def map_to_candidate(self, raw: RawObservation, t: int) -> Observation | None:
        """Map a heterogeneous raw observation to a candidate semantic trace.

        Returns None if the raw label can't be grounded in the shared
        vocabulary, or if it violates a domain/range constraint - the
        observation simply can't be integrated into collective cognition
        and is dropped.

        Grounding has two independent steps:
          - entity: raw.subject -> canonicalize_entity() -> canonical
            instance id (a no-op pass-through unless an alias is
            registered in entity_aliases - see its docstring above).
          - relation: raw.raw_relation -> relation_aliases[modality] ->
            canonical property name.

        Then branches on whether the canonical relation is an
        ObjectProperty or a DatatypeProperty (ObjectPropertySpec/
        DatatypePropertySpec above):
          - ObjectProperty: `domain` is checked against the SUBJECT's
            registered class, `range` against the OBJECT's (raw.raw_value is
            treated as another instance id, also passed through
            canonicalize_entity()) - both subclass-aware via
            class_hierarchy.is_a(). An unregistered subject or object is
            accepted (nothing to check it against), same policy either side.
          - DatatypeProperty: `domain` is checked the same way; `range` is a
            primitive type the raw value must parse as (e.g. float(raw_value)).

        This domain has no raw-label value aliasing (see value_aliases'
        docstring in __init__), so confidence here is just
        raw.observation_quality - there is no separate mapping-reliability
        factor to fold in, unlike a domain with abbreviation-style value
        ambiguity.
        """
        canonical_subject = canonicalize_entity(raw.subject)
        canonical_relation = self.relation_aliases.get(raw.modality, {}).get(raw.raw_relation)
        if canonical_relation is None:
            return None

        subject_class = self.subject_class.get(canonical_subject)

        if canonical_relation in self.object_properties:
            spec = self.object_properties[canonical_relation]
            if subject_class is not None and not self.class_hierarchy.is_a(subject_class, spec.domain):
                return None
            canonical_object = canonicalize_entity(raw.raw_value)
            object_class = self.subject_class.get(canonical_object)
            if object_class is not None and not self.class_hierarchy.is_a(object_class, spec.range):
                return None
        elif canonical_relation in self.datatype_properties:
            spec = self.datatype_properties[canonical_relation]
            if subject_class is not None and not self.class_hierarchy.is_a(subject_class, spec.domain):
                return None
            try:
                spec.range(raw.raw_value)
            except (TypeError, ValueError):
                return None
            canonical_object = raw.raw_value
        else:
            return None

        return Observation(
            subject=canonical_subject,
            relation=canonical_relation,
            object=canonical_object,
            location=raw.location,
            obs_time=t,
            confidence=raw.observation_quality,
            agent_id=raw.agent_id,
        )

    def infer_with_provenance(
        self, edges: dict[tuple[str, str], FieldEdge]
    ) -> list[tuple[Observation, ReasoningRule, list[FieldEdge]]]:
        """Evaluate every reasoning_rules entry against the field's current
        edges (M2's generalized join - see Atom/ReasoningRule above). Returns
        one candidate trace per satisfying variable binding, together with
        the rule and the premise edges that binding used."""
        facts_by_relation: dict[str, list[FieldEdge]] = {}
        for edge in edges.values():
            facts_by_relation.setdefault(edge.relation, []).append(edge)

        out = []
        for rule in self.reasoning_rules:
            # which antecedent atom shares the consequent's subject variable/
            # constant, if any - the inferred edge should be located where
            # the antecedent about the SAME subject as the consequent is,
            # not an arbitrary premise (defaulting to premises[0] could pick
            # a premise about a DIFFERENT subject entirely, e.g. a context/
            # co-location fact, and place the inferred edge somewhere
            # unrelated to what it's actually about).
            consequent_subject_idx = next(
                (i for i, atom in enumerate(rule.antecedents) if atom.subject == rule.consequent.subject),
                0,
            )
            for bindings, premises in _solve_rule(list(rule.antecedents), {}, [], facts_by_relation):
                if not self._satisfies_type_constraints(rule.type_constraints, bindings):
                    continue

                def resolve(token: str) -> str:
                    return bindings[token] if _is_variable(token) else token

                candidate = Observation(
                    subject=resolve(rule.consequent.subject),
                    relation=rule.consequent.relation,
                    object=resolve(rule.consequent.object),
                    location=premises[consequent_subject_idx].location,
                    obs_time=max(p.last_observation_time for p in premises),
                    # design memo: highest confidence among conditions, not lowest
                    confidence=max(p.confidence for p in premises),
                    agent_id=f"inference:{rule.name}",
                )
                out.append((candidate, rule, premises))
        return out

    def _satisfies_type_constraints(self, type_constraints: dict, bindings: dict) -> bool:
        for var, required_class in type_constraints.items():
            # type_constraints keys are bare variable names ("h"), matching
            # the design memo's YAML - _solve_rule's bindings dict keys
            # carry the Atom token's own "?" prefix ("?h"), so look up
            # accordingly rather than by the bare name (a mismatch here
            # would silently skip every check instead of failing loudly).
            value = bindings.get(f"?{var}")
            if value is None:
                continue  # not a variable this rule's antecedents actually bound - nothing to check
            value_class = self.subject_class.get(value)
            if value_class is None or not self.class_hierarchy.is_a(value_class, required_class):
                return False
        return True

    def infer(self, edges: dict[tuple[str, str], FieldEdge]) -> list[Observation]:
        """Like infer_with_provenance(), but returns just the candidates."""
        return [cand for cand, _, _ in self.infer_with_provenance(edges)]


# ============================================================
# VALIDATION HELPERS
# ============================================================
# Each takes field_sim.field.Field.edges (dict[storage key -> FieldEdge],
# see field_sim.cardinality.edge_key) and returns a list of violation
# descriptions - empty list means "no violations found", not "not checked".

def _has_triple(edges: dict, subject: str, relation: str, obj: str) -> bool:
    edge = edges.get(edge_key(subject, relation, obj))
    return edge is not None and edge.object == obj


def check_symmetric_property(edges: dict, relation: str) -> list[tuple[str, str]]:
    """Every (a, relation, b) edge must have a matching (b, relation, a)
    edge. Only meaningful for a relation declared symmetric; returns []
    unchecked (not an error) for one that isn't."""
    if not get_property_characteristics(relation).get("symmetric"):
        return []
    return [
        (edge.subject, edge.object) for edge in edges.values()
        if edge.relation == relation and not _has_triple(edges, edge.object, relation, edge.subject)
    ]


def check_irreflexive_property(edges: dict, relation: str) -> list[str]:
    """No (subject, relation, subject) edge - a relation can't hold between
    an entity and itself."""
    if not get_property_characteristics(relation).get("irreflexive"):
        return []
    return [edge.subject for edge in edges.values() if edge.relation == relation and edge.object == edge.subject]


def check_disjoint_properties(edges: dict, relation: str) -> list[tuple[str, str]]:
    """ABox/live-data check: the same subject can't hold both `relation`
    and one of its declared disjointWith relations TOWARD THE SAME OBJECT
    at once (e.g. Humanoid_1 movingToward Area_A and Humanoid_1
    movingAwayFrom Area_A simultaneously is a contradiction; movingToward
    Area_A and movingAwayFrom Corridor_1 is not - different targets, no
    conflict). Returns [] unchecked (not an error) for a relation with no
    declared disjointWith."""
    disjoint_with = get_property_relationships(relation).get("disjointWith", [])
    if not disjoint_with:
        return []
    return [
        (edge.subject, edge.object) for edge in edges.values()
        if edge.relation == relation
        for other_relation in disjoint_with
        if _has_triple(edges, edge.subject, other_relation, edge.object)
    ]


def check_class_disjointness(
    class_hierarchy: ClassHierarchy, disjoint: dict[str, list[str]] | None = None
) -> list[tuple[str, str]]:
    """TBox/schema check: no two classes declared disjointWith each other
    (class_disjoint_with, or an explicit `disjoint` override for testing)
    may also be in an ancestor/descendant relationship in `class_hierarchy`
    - that would mean every instance of the descendant structurally IS_A
    both, contradicting the disjointness declaration. This checks the
    DECLARATION against the hierarchy, not any instance data (see this
    module's docstring / class_disjoint_with's docstring for why a live,
    per-instance version would be structurally vacuous) - so unlike
    check_functional_property (removed, D2), this genuinely can fail: feed
    it a hierarchy/declaration pair where a "disjoint" class was
    accidentally reparented under the other, and it reports the conflict."""
    disjoint = class_disjoint_with if disjoint is None else disjoint
    violations = []
    for cls, others in disjoint.items():
        for other in others:
            if class_hierarchy.is_a(cls, other) or class_hierarchy.is_a(other, cls):
                violations.append((cls, other))
    return violations
