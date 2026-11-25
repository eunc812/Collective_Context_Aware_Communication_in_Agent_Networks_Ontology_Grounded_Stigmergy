"""Ground-truth environment: the VLA humanoid domain's total set of
(subject, relation, object) triples - one Fact per (instance, applicable
property) pair, generated entirely from a field_sim.ontology.
KnowledgeBase's Class/Instance/Property schema rather than a hardcoded
vocabulary. Each fact independently has a chance to change to a new valid
value every round, modeling a continuously changing environment.

Agents observe only a spatially-local SUBSET of this total set
(Agent.observation_radius, field_sim.agent.Agent.observe()), and keep only a
further subset of THAT as their own local_cognition - the environment is the
full population of facts an agent's partial observation is a subset of.
"""

import math
import random

from field_sim.ontology import KnowledgeBase
from field_sim.types import Bounds, Fact, Location, resolve_area

# The Instance layer example from the design memo.
DEFAULT_INSTANCES: dict[str, str] = {
    "Humanoid_1": "Humanoid", "Humanoid_2": "Humanoid",
    "Ped_1": "Pedestrian", "Ped_2": "Pedestrian",
    "Box_1": "Package", "Box_2": "Package",
    "Chair_1": "Furniture", "Door_1": "Door",
    "Area_A": "Area", "Corridor_1": "Corridor", "Room_1": "Room",
}

# Instance-count defaults for generate_instances(). Area defaults to
# max(2, num_agents // 4) rather than a fixed number: R1 (approaching_area)
# and R4 (potential_conflict) both require their "a" binding to be
# class-exactly Area (type_constraints={"a": "Area"}, not the broader
# SpatialRegion) - with only ONE Area instance (DEFAULT_INSTANCES' Area_A),
# every approachingArea/potentialConflict/shouldSlowDown conclusion R1/R4/R5
# can ever produce funnels through that single node no matter how many
# Humanoids/Pedestrians exist. Scaling Area count roughly with agent count
# keeps that funnel from becoming the dominant bottleneck as the population
# grows. Room/Corridor/WorkZone/Passage don't have this constraint (R5's
# "a" is plain SpatialRegion) so they scale on a flatter, fixed-ish curve.
_LEAF_CLASS_PREFIX = {
    "Humanoid": "Humanoid", "Pedestrian": "Ped",
    "Package": "Box", "Container": "Container", "Tool": "Tool",
    "Furniture": "Chair", "Wall": "Wall", "Door": "Door", "Obstacle": "Obstacle",
    "Area": "Area", "Room": "Room", "Corridor": "Corridor",
    "WorkZone": "WorkZone", "Passage": "Passage",
}

# Proximity cutoff for carrying/interactingWith's candidate pool (a Humanoid
# can only start carrying/interacting with something currently within this
# distance) - see Environment._new_object_fact/step for where it's used.
# Not used by `near` itself, which always picks the SINGLE closest entity
# regardless of distance (no cutoff needed - there's always a closest one).
NEAR_THRESHOLD = 15.0


def generate_instances(
    num_humanoids: int = 4,
    num_pedestrians: int = 6,
    num_packages: int = 4,
    num_containers: int = 2,
    num_tools: int = 2,
    num_furniture: int = 3,
    num_walls: int = 2,
    num_doors: int = 3,
    num_obstacles: int = 2,
    num_areas: int | None = None,
    num_rooms: int = 4,
    num_corridors: int = 3,
    num_workzones: int = 2,
    num_passages: int = 2,
) -> dict[str, str]:
    """Procedurally build an `instances` dict of the same shape as
    DEFAULT_INSTANCES (subject id -> class), for any population size -
    Environment(kb, instances=...) accepts either interchangeably, no
    other code needs to change.

    Spread across every LEAF class the reasoning rules (R0-R6) actually
    touch as an antecedent/consequent type - not just Humanoid/Pedestrian
    count - since R2/R3 (path_blocked/should_reroute) need PhysicalObject
    instances to serve as `blocks` material, and R1/R4 need enough Area
    instances specifically (see _LEAF_CLASS_PREFIX's docstring above) or
    every conflict conclusion keeps funneling through one node regardless
    of how large the agent population gets.
    """
    if num_areas is None:
        num_areas = max(2, (num_humanoids + num_pedestrians) // 4)
    counts = {
        "Humanoid": num_humanoids, "Pedestrian": num_pedestrians,
        "Package": num_packages, "Container": num_containers, "Tool": num_tools,
        "Furniture": num_furniture, "Wall": num_walls, "Door": num_doors, "Obstacle": num_obstacles,
        "Area": num_areas, "Room": num_rooms, "Corridor": num_corridors,
        "WorkZone": num_workzones, "Passage": num_passages,
    }
    instances: dict[str, str] = {}
    for cls, n in counts.items():
        prefix = _LEAF_CLASS_PREFIX[cls]
        for i in range(1, n + 1):
            instances[f"{prefix}_{i}"] = cls
    return instances


def generate_instances_matched_to_regions(
    num_areas: int = 3,
    num_rooms: int = 4,
    num_corridors: int = 3,
    num_workzones: int = 2,
    num_passages: int = 2,
    num_pedestrians: int = 6,
    fill_grid: bool = True,
    **other_leaf_counts,
) -> dict[str, str]:
    """generate_instances(), but num_humanoids is forced to exactly equal
    the total SpatialRegion instance count - for pairing with
    place_humanoids_at_region_centers() below, which needs a 1:1 Humanoid
    <-> region mapping to place one Humanoid at each region's center.

    fill_grid=True (the default): _partition_regions() itself is gapless
    for any region count now (a partially-filled last row just gets wider
    cells), so this is no longer about avoiding unnamed gaps - it's purely
    about VISUAL regularity for a uniform, one-region-per-cell grid (see
    figures/make_environment_overview.py): _partition_regions' ceil(sqrt(N))
    grid has cols*rows cells, which only equals N (num_areas+num_rooms+
    num_corridors+num_workzones+num_passages) when N is already a perfect
    square - otherwise the last row has fewer, wider cells than the rest.
    Padding num_passages up so the region count exactly equals the grid's
    capacity (cols*rows) makes every cell the same size - set False to get
    the old, possibly-uneven-last-row count instead (still gapless either
    way).

    `other_leaf_counts` passes straight through to generate_instances()
    (num_packages, num_containers, num_tools, num_walls, num_doors,
    num_obstacles) - everything except num_humanoids/num_areas/num_passages,
    which this function itself controls.
    """
    total_regions = num_areas + num_rooms + num_corridors + num_workzones + num_passages
    if fill_grid:
        cols = math.ceil(math.sqrt(total_regions))
        rows = math.ceil(total_regions / cols)
        capacity = cols * rows
        num_passages += capacity - total_regions  # pad the trailing class to fill every grid cell
        total_regions = capacity
    return generate_instances(
        num_humanoids=total_regions, num_pedestrians=num_pedestrians,
        num_areas=num_areas, num_rooms=num_rooms, num_corridors=num_corridors,
        num_workzones=num_workzones, num_passages=num_passages,
        **other_leaf_counts,
    )


def _partition_regions(
    region_instances: list[str], area_size: float,
) -> tuple[dict[str, Bounds], dict[str, tuple[int, int]]]:
    """Divide the area_size x area_size space into one non-overlapping,
    axis-aligned rectangle per SpatialRegion instance (a simple ceil(sqrt(N))
    grid - not a spec of any real building layout, just enough structure
    that "which rectangle contains this point" is well-defined). This is
    what makes an entity's continuous location and its discrete locatedAt
    area agree with each other, instead of location being a uniform-random
    point unrelated to which area the entity is in.

    Gapless for ANY n, not just n = cols*rows exactly: ceil(sqrt(n))
    columns and ceil(n/cols) rows still set the grid's shape, but each
    row's cell width is area_size / (however many regions actually land in
    THAT row), not always area_size/cols - so a partially-filled last row
    (n not a multiple of cols) still spans x=[0, area_size] with wider
    cells instead of leaving the remaining columns' worth of space
    unassigned to any region. Every point in [0, area_size]^2 therefore
    resolves to exactly one region (field_sim.decision.resolve_area never
    returns None for an in-bounds location) without needing to pad the
    region count up to a perfect rectangle first (contrast
    generate_instances_matched_to_regions(fill_grid=True) below, which
    pads for a different reason - uniform, one-region-per-cell visual
    regularity, not gap-avoidance).

    Also returns each region's (row, col) grid index alongside its bounds -
    Environment._region_neighbors_3x3() uses this to find a region's
    immediate grid neighbors without having to infer them back out of raw
    rectangle coordinates (which row heights/col widths vary between rows
    for a non-perfect-square n)."""
    n = len(region_instances)
    if n == 0:
        return {}, {}
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_h = area_size / rows
    bounds = {}
    grid_pos = {}
    i = 0
    for row in range(rows):
        cells_in_row = min(cols, n - i)
        cell_w = area_size / cells_in_row
        for col in range(cells_in_row):
            region = region_instances[i]
            bounds[region] = (col * cell_w, (col + 1) * cell_w, row * cell_h, (row + 1) * cell_h)
            grid_pos[region] = (row, col)
            i += 1
    return bounds, grid_pos


def _bbox_center(bounds: Bounds) -> Location:
    x_min, x_max, y_min, y_max = bounds
    return Location((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _sample_in_bounds(rng: random.Random, bounds: Bounds) -> Location:
    x_min, x_max, y_min, y_max = bounds
    return Location(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))


class Environment:
    """VLA humanoid domain's dynamic ground truth.

    Which properties a subject can have is entirely schema-driven: for every
    non-inferred ObjectProperty/DatatypeProperty in `kb`, every instance
    whose class satisfies that property's `domain` (subclass-aware, via
    kb.class_hierarchy) gets one fact for it. Inferred-only properties
    (ObjectPropertySpec.inferred=True - approachingArea, pathBlocked, ...)
    are never generated here; they only ever come from field_sim.field.
    Field.reason() applying kb.reasoning_rules.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        instances: dict[str, str] | None = None,
        area_size: float = 100.0,
        change_prob: float = 0.05,
        seed: int | None = None,
        agent_move_step: float = 6.0,
    ):
        self.kb = kb
        self.rng = random.Random(seed)
        self.area_size = area_size
        self.change_prob = change_prob
        # Default 6.0: the FIXED magnitude of a drift step (max uniform
        # per-axis offset) for ontology-"Agent"-class instances (Humanoid/
        # Pedestrian/...), which physically drift a random step ON ROUNDS
        # `change_prob` PERMITS (see step() below) instead of having their
        # locatedAt value periodically reassigned - see _drift_subject()'s
        # docstring for how locatedAt is DERIVED from position here, the
        # reverse of every other subject's position-follows-locatedAt rule.
        # `change_prob` alone now controls HOW OFTEN this fires (2026-09-10:
        # unified with every other kind of change in this class instead of
        # drifting unconditionally every round), `agent_move_step` only ever
        # controls HOW FAR a step goes when it does. Pass 0.0 explicitly to
        # get the old jump-only behavior instead (no drift capability at
        # all - see test_agent_move_step_zero_keeps_old_jump_based_behavior).
        self.agent_move_step = agent_move_step

        self.subject_classes: dict[str, str] = dict(instances or DEFAULT_INSTANCES)
        for subject, cls in self.subject_classes.items():
            kb.register_subject(subject, cls)

        # Partition the space into one rectangle per SpatialRegion instance
        # (Area_A, Corridor_1, Room_1, ...) - each such instance occupies
        # its own sub-region of the overall area_size x area_size space,
        # rather than every instance getting an independent uniform-random
        # point unrelated to which area it's actually in.
        self.region_instances: list[str] = [
            subject for subject, cls in self.subject_classes.items()
            if self.kb.class_hierarchy.is_a(cls, "SpatialRegion")
        ]
        self.region_bounds: dict[str, Bounds]
        self.region_grid_pos: dict[str, tuple[int, int]]
        self.region_bounds, self.region_grid_pos = _partition_regions(self.region_instances, area_size)

        self.locations: dict[str, Location] = {}
        for region in self.region_instances:
            self.locations[region] = _bbox_center(self.region_bounds[region])

        # Every non-region entity's initial area (its locatedAt value) is
        # chosen up front, together with a location sampled inside that
        # area's rectangle - established together so the two can never
        # disagree, unlike a location assigned independently of any area.
        self._located_at: dict[str, str] = {}
        locatedat_spec = self.kb.object_properties.get("locatedAt")
        if locatedat_spec is not None and self.region_instances:
            for subject, cls in self.subject_classes.items():
                if subject in self.region_bounds:
                    continue
                if not self.kb.class_hierarchy.is_a(cls, locatedat_spec.domain):
                    continue
                area = self.rng.choice(self.region_instances)
                self._located_at[subject] = area
                self.locations[subject] = _sample_in_bounds(self.rng, self.region_bounds[area])

        # Any subject that got neither a region rectangle nor a locatedAt
        # area (no region instances exist at all, or this KB has no
        # locatedAt property) falls back to a uniform-random point over
        # the whole space, same as before this partitioning existed.
        for subject in self.subject_classes:
            if subject not in self.locations:
                self.locations[subject] = Location(self.rng.uniform(0, area_size), self.rng.uniform(0, area_size))

        # Subjects ELIGIBLE to drift (see _drift_subject) - ontology-"Agent"-
        # class instances only (Humanoid/Human/Pedestrian - the classes
        # hasSpeed/movingToward/movingAwayFrom are actually defined for), and
        # only when agent_move_step > 0 was requested (a magnitude has to be
        # configured for drifting to mean anything); an instance with no
        # established locatedAt area (see above) has nothing for
        # _drift_subject to derive, so it's excluded too. Whether an eligible
        # subject actually drifts on any given round is decided by
        # `change_prob` in step() below, same as every other kind of change
        # in this class - membership here is just "capable of it".
        self._drifting_subjects: set[str] = {
            subject for subject in self._located_at
            if agent_move_step > 0 and self.kb.class_hierarchy.is_a(self.subject_classes[subject], "Agent")
        }

        self.facts: dict[tuple[str, str], Fact] = {}
        for relation, spec in kb.object_properties.items():
            if spec.inferred:
                continue
            for subject in self.subject_classes:
                fact = self._new_object_fact(subject, relation)
                if fact is not None:
                    self.facts[fact.key] = fact
        for relation, spec in kb.datatype_properties.items():
            for subject in self.subject_classes:
                fact = self._new_datatype_fact(subject, relation)
                if fact is not None:
                    self.facts[fact.key] = fact

    def _range_candidates(self, relation: str, exclude: str) -> list[str]:
        spec = self.kb.object_properties[relation]
        return [
            subject for subject, cls in self.subject_classes.items()
            if subject != exclude and self.kb.class_hierarchy.is_a(cls, spec.range)
        ]

    def _nearest_entity(self, subject: str, relation: str) -> str | None:
        """Ground truth for `near`: the closest OTHER valid-range subject to
        `subject`'s CURRENT location, by real distance - not randomized.
        None only if no valid candidate exists at all (mirrors
        _range_candidates returning []).

        This is the OBSERVABLE near fact - one per subject, what a sensor
        reports. Nearest-neighbour is not symmetric (A's nearest can be B
        while B's nearest is C), whereas the ontology declares near
        symmetric; the full set of valid near triples is therefore the
        symmetric closure of these facts, which near is multi-valued to
        hold (field_sim.cardinality) and which
        field_sim.evaluation.reference_state() builds via the
        near_symmetric rule."""
        candidates = self._range_candidates(relation, exclude=subject)
        if not candidates:
            return None
        loc = self.locations[subject]
        return min(candidates, key=lambda c: loc.distance_to(self.locations[c]))

    def _region_neighbors_3x3(self, region: str) -> set[str]:
        """`region` plus every region within one grid step of it (row and
        col both within +-1, clipped at the grid's edges by construction -
        no explicit bounds check needed since a nonexistent (row+-1, col+-1)
        simply matches nothing). Used to keep movingToward/movingAwayFrom
        local (see _aligned_region) instead of ranging over the entire
        space."""
        row, col = self.region_grid_pos[region]
        return {
            r for r, (rr, rc) in self.region_grid_pos.items()
            if abs(rr - row) <= 1 and abs(rc - col) <= 1
        }

    def _aligned_region(
        self, subject: str, dx: float, dy: float, relation: str, want_max: bool,
        exclude_value: str | None = None,
    ) -> str | None:
        """Ground truth for movingToward (want_max=True) / movingAwayFrom
        (want_max=False): the region whose center is most (or least)
        aligned with the (dx, dy) velocity direction, by dot product
        against the direction-to-center vector. `relation` only selects
        which property's range-class candidates are eligible (movingToward
        and movingAwayFrom share the same domain/range, so either name
        yields the same candidate list - passed through for clarity/
        future-proofing rather than hardcoding one).

        Candidates are first restricted to the 3x3 grid neighborhood
        (_region_neighbors_3x3) around the subject's OWN current region,
        before ranking by alignment - a pure global dot-product could pick
        a region clear across the map just because its direction happens to
        line up best, even though the subject (moving agent_move_step per
        round) will realistically never get there soon; anchoring to nearby
        regions first keeps "moving toward area X" meaning a plausible
        near-term destination. Falls back to the full candidate list if the
        subject's own region can't be resolved (no SpatialRegion instances
        at all) or the neighborhood happens to contain no valid candidate.

        `exclude_value` enforces movingToward/movingAwayFrom's declared
        property-disjointness (property_relationships' "disjointWith" -
        the same subject can't be moving toward AND away from the SAME
        region at once): _drift_subject passes movingToward's just-computed
        result here when calling this for movingAwayFrom, so the min-search
        below can never land back on it. Without this, the two are only
        NOT equal because max and min happen to disagree over the alignment
        scores - true for essentially every real (dx, dy) (an exact tie
        needs alignment scores equal to full float precision, empirically
        never observed across 188,800 sampled rounds/subjects), but nothing
        stopped it structurally the way the other property/class
        constraints in this codebase are - see field_sim.ontology's
        check_disjoint_properties(), previously enforced nowhere outside
        tests. Skipped (falls back to the unfiltered candidate list) if
        removing it would leave no candidates at all - a real ground-truth
        value is still preferable to no fact, and the only way this can
        even arise is the single-candidate edge case where nothing else is
        available to point at anyway."""
        candidates = self._range_candidates(relation, exclude=subject)
        if not candidates:
            return None
        loc = self.locations[subject]

        own_region = resolve_area(loc, self.region_bounds)
        if own_region is not None and own_region in self.region_grid_pos:
            nearby = self._region_neighbors_3x3(own_region)
            local_candidates = [c for c in candidates if c in nearby]
            if local_candidates:
                candidates = local_candidates

        if exclude_value is not None:
            candidates = [c for c in candidates if c != exclude_value] or candidates

        def alignment(region: str) -> float:
            c = self.locations[region]
            return (c.x - loc.x) * dx + (c.y - loc.y) * dy

        return (max if want_max else min)(candidates, key=alignment)

    def _sample_datatype_value(self, spec) -> str:
        if spec.range is float:
            return str(round(self.rng.uniform(0.0, 2.0), 2))
        return str(self.rng.random())

    def _new_object_fact(self, subject: str, relation: str) -> Fact | None:
        spec = self.kb.object_properties[relation]
        if not self.kb.class_hierarchy.is_a(self.subject_classes[subject], spec.domain):
            # A SpatialRegion instance now fails this check for locatedAt
            # specifically (locatedAt's domain is PhysicalEntity = Agent +
            # PhysicalObject, deliberately excluding SpatialRegion - see
            # field_sim.ontology.KnowledgeBase's PhysicalEntity/locatedAt
            # comments): a region has no meaningful locatedAt of its own
            # (all regions are same-level grid cells, no real containment),
            # so it simply gets no locatedAt fact at all, instead of a
            # vacuous self-referential one like "Area_1 locatedAt Area_2".
            return None
        if relation == "locatedAt" and subject in self._located_at:
            # Reuse the area __init__ already committed to when it sampled
            # this subject's location inside that area's rectangle - a
            # fresh independent random choice here could disagree with it.
            obj = self._located_at[subject]
        elif relation == "near":
            # Geometric from t=0 - always well-defined (no velocity needed,
            # just current position), so no random fallback like
            # movingToward/movingAwayFrom below need before anyone's moved.
            obj = self._nearest_entity(subject, relation)
            if obj is None:
                return None
        elif relation == "blocks":
            # Geometric from t=0 - whatever region already contains this
            # object's initial location.
            obj = resolve_area(self.locations[subject], self.region_bounds)
            if obj is None:
                return None
        elif relation in ("carrying", "interactingWith"):
            # Prefer a candidate within NEAR_THRESHOLD (a Humanoid shouldn't
            # start out "carrying" something on the other side of the map);
            # fall back to the full candidate list if nothing happens to be
            # nearby yet, so this fact still always gets created like every
            # other relation's does, rather than silently going missing for
            # some subjects based on their random initial placement.
            candidates = self._range_candidates(relation, exclude=subject)
            loc = self.locations[subject]
            nearby = [c for c in candidates if loc.distance_to(self.locations[c]) <= NEAR_THRESHOLD]
            candidates = nearby or candidates
            if not candidates:
                return None
            obj = self.rng.choice(candidates)
        else:
            # movingToward/movingAwayFrom (no velocity yet at t=0 - real
            # drift hasn't happened, so there's nothing geometric to derive
            # yet; _drift_subject takes over from the first step() onward
            # for drifting subjects) and locatedAt for anything NOT already
            # resolved above (region instances, non-drifting subjects).
            candidates = self._range_candidates(relation, exclude=subject)
            if not candidates:
                return None
            obj = self.rng.choice(candidates)
        return Fact(subject, relation, obj, self.locations[subject], last_changed_time=0)

    def _new_datatype_fact(self, subject: str, relation: str) -> Fact | None:
        spec = self.kb.datatype_properties[relation]
        if not self.kb.class_hierarchy.is_a(self.subject_classes[subject], spec.domain):
            return None
        return Fact(subject, relation, self._sample_datatype_value(spec),
                    self.locations[subject], last_changed_time=0)

    def step(self, t: int) -> list[Fact]:
        """Advance the environment by one tick. Returns facts that changed this step."""
        changed = []
        # sorted(), not a bare set iteration: self._drifting_subjects is a
        # set, whose iteration order depends on str hash values - randomized
        # per PROCESS (PYTHONHASHSEED) unless pinned, not just per `seed`.
        # Each subject's dice roll below consumes one self.rng.random() draw
        # in loop order, so an unstable iteration order silently reassigns
        # WHICH subject gets which draw from an otherwise fully seeded RNG -
        # the same cfg.seed could then still produce a different run end to
        # end, purely from which Python process happened to run it (verified:
        # re-running the identical ExperimentConfig in 3 separate `python -c`
        # invocations gave 3 different ccg/cost results before this fix).
        for subject in sorted(self._drifting_subjects):
            # Gated by change_prob, same as every other kind of change below
            # (2026-09-10: unified "environment speed" into this single
            # knob - a drift-eligible subject no longer moves every round
            # unconditionally, it moves on whichever rounds this roll
            # succeeds, exactly like carrying/interactingWith/non-drifting
            # locatedAt already worked).
            if self.rng.random() < self.change_prob:
                self._drift_subject(subject, t, changed)

        for fact in self.facts.values():
            if fact.subject in self._drifting_subjects and fact.relation in (
                "locatedAt", "movingToward", "movingAwayFrom", "hasSpeed",
            ):
                continue  # driven by _drift_subject above, not this method

            if fact.relation == "near":
                # Live geometric recompute every round - `near`'s ground
                # truth is a pure function of CURRENT position, no change_
                # prob roll of its own. It still only changes VALUE when the
                # underlying position actually moved, which for a drifting
                # subject is itself change_prob-gated up above (this recompute
                # just picks up whatever position resulted, unconditionally).
                new_obj = self._nearest_entity(fact.subject, "near")
                if new_obj is not None and new_obj != fact.object:
                    fact.object, fact.last_changed_time = new_obj, t
                    changed.append(fact)
                continue

            if fact.relation == "blocks":
                # Live geometric recompute - whatever region currently
                # contains this object's position. PhysicalObjects don't
                # move on their own, so this only actually changes if the
                # object's own location changes via the still-change_prob-
                # gated locatedAt path below (for a non-drifting subject).
                new_obj = resolve_area(self.locations[fact.subject], self.region_bounds)
                if new_obj is not None and new_obj != fact.object:
                    fact.object, fact.last_changed_time = new_obj, t
                    changed.append(fact)
                continue

            if self.rng.random() >= self.change_prob:
                continue

            if fact.relation in ("carrying", "interactingWith"):
                # Discrete pick-up/interact events, not a continuous
                # geometric quantity (unlike near) - still change_prob-
                # gated for WHEN they change, but the new value can only
                # come from something currently within NEAR_THRESHOLD.
                loc = self.locations[fact.subject]
                candidates = [
                    c for c in self._range_candidates(fact.relation, exclude=fact.subject)
                    if loc.distance_to(self.locations[c]) <= NEAR_THRESHOLD
                ]
                if not candidates:
                    continue  # nothing eligible nearby this round - keep the existing value
                new_obj = self.rng.choice([c for c in candidates if c != fact.object] or candidates)
                if new_obj != fact.object:
                    fact.object, fact.last_changed_time = new_obj, t
                    changed.append(fact)
                continue

            if fact.relation in self.kb.object_properties:
                # Whatever's left: locatedAt for non-drifting NON-region
                # subjects (a region instance never gets a locatedAt fact
                # at all - see _new_object_fact's docstring for why), and
                # movingToward/movingAwayFrom for non-drifting Agent-class
                # subjects - no real velocity to derive them from, so they
                # keep the original random-reassignment behavior.
                candidates = self._range_candidates(fact.relation, exclude=fact.subject)
                candidates = [c for c in candidates if c != fact.object] or candidates
                if not candidates:
                    continue
                fact.object = self.rng.choice(candidates)
                if fact.relation == "locatedAt" and fact.object in self.region_bounds:
                    self._move_subject(fact.subject, fact.object)
            else:
                fact.object = self._sample_datatype_value(self.kb.datatype_properties[fact.relation])
            fact.last_changed_time = t
            changed.append(fact)
        return changed

    def _move_subject(self, subject: str, area: str) -> None:
        """Resample `subject`'s location inside `area`'s rectangle, and
        propagate the new point to every one of its facts - all of a
        subject's facts share one physical position, so a locatedAt change
        has to move them together, not just the locatedAt fact itself."""
        new_location = _sample_in_bounds(self.rng, self.region_bounds[area])
        self.locations[subject] = new_location
        self._located_at[subject] = area
        for fact in self.facts.values():
            if fact.subject == subject:
                fact.location = new_location

    def _drift_subject(self, subject: str, t: int, changed: list[Fact]) -> None:
        """One random-direction movement step for an opt-in "Agent"-class
        instance (agent_move_step > 0), called from step() on whichever
        rounds its change_prob roll succeeds (not every round - see step())
        - the REVERSE of _move_subject's
        causality: everywhere else in this class, a symbolic locatedAt
        change drives where a subject's location gets resampled to; here,
        the physical position drifts on its own every round (a plain random
        walk, like field_sim.agent.Agent.move() - direction is not
        purposeful), and locatedAt is DERIVED from wherever that walk
        lands, only writing a change (and appending to `changed`) when the
        new position actually crossed into a different region's rectangle.

        Still keeps location and locatedAt in agreement at every moment,
        same invariant _move_subject maintains for every other subject -
        just arrived at from the opposite direction. The `new_area is None`
        guard below is now effectively unreachable in practice: `new_x`/
        `new_y` are already clamped into [0, area_size]^2, and
        _partition_regions() gaplessly tiles that entire square for any
        nonzero region count - so every clamped step lands in a named
        region. Kept as a defensive fallback (reject and stay put rather
        than commit a location no named area agrees with) in case
        `region_bounds` is ever constructed some other way that doesn't
        gaplessly tile the space.

        hasSpeed/movingToward/movingAwayFrom are derived from this SAME
        (dx, dy) step - the real distance moved, and whichever region the
        real velocity vector points most toward/away from
        (Environment._aligned_region) - instead of being independently
        randomized like every other relation. If the step is rejected
        (new_area is None) these aren't touched either - nothing physically
        moved, so nothing here should appear to have changed."""
        loc = self.locations[subject]
        dx = self.rng.uniform(-self.agent_move_step, self.agent_move_step)
        dy = self.rng.uniform(-self.agent_move_step, self.agent_move_step)
        new_x = min(max(loc.x + dx, 0.0), self.area_size)
        new_y = min(max(loc.y + dy, 0.0), self.area_size)
        new_location = Location(new_x, new_y)

        new_area = resolve_area(new_location, self.region_bounds)
        if new_area is None:
            return  # would leave every named region's rectangle - reject, stay put

        self.locations[subject] = new_location
        for fact in self.facts.values():
            if fact.subject == subject:
                fact.location = new_location

        locatedat_fact = self.facts.get((subject, "locatedAt"))
        if locatedat_fact is not None and locatedat_fact.object != new_area:
            locatedat_fact.object = new_area
            locatedat_fact.last_changed_time = t
            self._located_at[subject] = new_area
            changed.append(locatedat_fact)

        speed_fact = self.facts.get((subject, "hasSpeed"))
        if speed_fact is not None:
            speed_str = str(round(math.sqrt(dx * dx + dy * dy), 2))
            if speed_fact.object != speed_str:
                speed_fact.object = speed_str
                speed_fact.last_changed_time = t
                changed.append(speed_fact)

        # movingToward computed first, unconditionally, so its value is
        # available to exclude from movingAwayFrom's own search below even
        # when this subject has no movingToward fact of its own (an edge
        # case that doesn't arise in practice - both share domain=Agent -
        # but this keeps the disjointness guard correct either way).
        toward = self._aligned_region(subject, dx, dy, "movingToward", want_max=True)
        toward_fact = self.facts.get((subject, "movingToward"))
        if toward_fact is not None and toward is not None and toward_fact.object != toward:
            toward_fact.object = toward
            toward_fact.last_changed_time = t
            changed.append(toward_fact)

        away_fact = self.facts.get((subject, "movingAwayFrom"))
        if away_fact is not None:
            away = self._aligned_region(subject, dx, dy, "movingAwayFrom", want_max=False, exclude_value=toward)
            if away is not None and away_fact.object != away:
                away_fact.object = away
                away_fact.last_changed_time = t
                changed.append(away_fact)


def place_humanoids_at_region_centers(environment: "Environment", jitter: float = 0.0) -> None:
    """Post-construction repositioning: assign each Humanoid instance to a
    DISTINCT region (one-to-one, matched by sorted id) and place it at that
    region's center (jitter=0.0, the default) or a small random offset from
    it (jitter>0.0: +/- jitter on both x and y, clamped to stay inside that
    region's own rectangle so it can never cross into a neighboring one) -
    instead of Environment's default placement (a randomly-chosen area,
    then a uniform-random point anywhere within it). Requires exactly as
    many Humanoid instances as region instances (raises ValueError
    otherwise, since a 1:1 mapping wouldn't be possible) - see
    generate_instances_matched_to_regions() for building a population that
    satisfies this. Uses `environment.rng` (the Environment's own seeded
    generator) for the jitter, so this stays reproducible given the same
    Environment seed.

    Rewrites `environment.locations`/`environment._located_at` and every
    affected Humanoid's own facts (location on all of them, `object` on
    the locatedAt fact specifically) directly - Environment itself is
    otherwise unaware this happened, same as field_sim.environment.
    Environment._move_subject's own direct mutation.
    """
    humanoids = sorted(
        s for s, cls in environment.subject_classes.items()
        if environment.kb.class_hierarchy.is_a(cls, "Humanoid")
    )
    regions = sorted(environment.region_instances)
    if len(humanoids) != len(regions):
        raise ValueError(
            f"place_humanoids_at_region_centers needs num_humanoids == num_regions "
            f"(got {len(humanoids)} Humanoids, {len(regions)} regions)"
        )
    for humanoid, region in zip(humanoids, regions):
        x_min, x_max, y_min, y_max = environment.region_bounds[region]
        center = _bbox_center(environment.region_bounds[region])
        if jitter > 0.0:
            x = min(max(center.x + environment.rng.uniform(-jitter, jitter), x_min), x_max)
            y = min(max(center.y + environment.rng.uniform(-jitter, jitter), y_min), y_max)
            center = Location(x, y)
        environment.locations[humanoid] = center
        environment._located_at[humanoid] = region
        for fact in environment.facts.values():
            if fact.subject == humanoid:
                fact.location = center
                if fact.relation == "locatedAt":
                    fact.object = region
