"""Agents with partial, local observation of the environment, and the
population factory that builds them.

Humanoid-as-its-own-sensor (build_humanoid_linked_agents/
sync_agent_locations below) is the only sensor population this module
builds: every observer in the simulation IS a randomly-drifting Humanoid
VLA instance - there is no separate, VLA-independent sensor network (no
"a0"/"a1" style observers)."""

import random

from field_sim.communication import CommunicationPolicy, PolicyDecision
from field_sim.decision import Bounds, M3Result, m3_decide, resolve_area
from field_sim.environment import Environment
from field_sim.field import Field
from field_sim.ontology import KnowledgeBase
from field_sim.types import Location, Observation, RawObservation

MODALITIES = ["canonical", "numeric", "abbrev"]  # heterogeneous sensor vocabularies
RELIABILITY_RANGE = (0.6, 1.0)  # heterogeneous but decent sensors by default


class Agent:
    """An agent that observes only the environment within its local radius.

    This encodes the partial-observation assumption: no single agent has
    access to the full environment state.
    """

    def __init__(
        self,
        agent_id: str,
        location: Location,
        observation_radius: float = 20.0,
        reliability: float = 1.0,
        step_size: float = 0.0,
        modality: str = "canonical",
        seed: int | None = None,
    ):
        self.agent_id = agent_id
        self.location = location
        self.observation_radius = observation_radius
        self.step_size = step_size
        self.modality = modality  # which vocabulary/sensor encoding this agent reports in
        self.rng = random.Random(seed)

        # heterogeneous sensor quality: a cheap/degraded sensor reports
        # lower confidence on every reading. This is one contributor to the
        # field's single confidence number - the knowledge base's raw-label
        # mapping reliability is folded in alongside it in
        # map_to_candidate() (field_sim.ontology).
        self.reliability = max(0.0, min(1.0, reliability))

        # local cognition: the latest facts this agent believes to be true,
        # populated ONLY by this agent's own direct observation
        # (form_candidate_traces() below) - never overwritten by anything
        # read back from the field (see receive_field_context(), which
        # returns its M3 result without touching this at all).
        self.local_cognition: dict[tuple[str, str], Observation] = {}

    def move(self, area_size: float) -> None:
        """Random-walk the agent within [0, area_size]^2, reflecting off the edges.

        Movement is what makes decay observable: as an agent wanders off, its
        past observations stop being refreshed and the field trace evaporates.
        """
        if self.step_size <= 0:
            return
        dx = self.rng.uniform(-self.step_size, self.step_size)
        dy = self.rng.uniform(-self.step_size, self.step_size)
        x = min(max(self.location.x + dx, 0.0), area_size)
        y = min(max(self.location.y + dy, 0.0), area_size)
        self.location = Location(x, y)

    def observe(self, environment: Environment, kb: KnowledgeBase, t: int) -> list[RawObservation]:
        """Perceive facts within range and report them in this agent's own modality.

        Different agents may use different sensors/vocabularies for the same
        underlying fact (a numeric code vs. an abbreviation vs. plain text) -
        this is where observation heterogeneity comes from. What comes out of
        here is NOT yet a candidate semantic trace: it still needs to be
        grounded in the shared knowledge base via form_candidate_traces().

        observation_quality here reflects only this agent's sensor reliability
        (self.reliability) - it is NOT yet the field-facing confidence, which
        also folds in the knowledge base's raw-label mapping reliability once
        this raw reading gets grounded (field_sim.ontology.KnowledgeBase.
        map_to_candidate()). No sensor-glitch noise is injected here - every
        reading of a fact within range is reported faithfully, at this
        agent's own reliability.

        A fact whose subject IS this agent (fact.subject == self.agent_id -
        only possible when this Agent shares its id with a VLA instance,
        e.g. build_humanoid_linked_agents' Humanoid-as-sensor agents below)
        is always observed regardless of observation_radius - an
        agent always has information about itself, distance notwithstanding.
        For every other agent (an id no VLA fact is ever about, e.g. the
        default "a0"/"a1"/...) this is a no-op, same as before.
        """
        raws = []
        for fact in environment.facts.values():
            is_self = fact.subject == self.agent_id
            if not is_self and self.location.distance_to(fact.location) > self.observation_radius:
                continue

            encoded = kb.encode(self.modality, fact.relation, fact.object)
            if encoded is None:
                continue  # this agent's modality has no vocabulary for this relation
            raw_relation, raw_value = encoded

            raws.append(
                RawObservation(
                    subject=fact.subject,
                    modality=self.modality,
                    raw_relation=raw_relation,
                    raw_value=raw_value,
                    location=fact.location,
                    obs_time=t,
                    observation_quality=self.reliability,
                    agent_id=self.agent_id,
                )
            )
        return raws

    def form_candidate_traces(
        self, raws: list[RawObservation], kb: KnowledgeBase, t: int
    ) -> list[Observation]:
        """Ground raw observations in the shared vocabulary; unmappable ones are dropped.

        This is where local cognition is actually formed: each raw reading
        becomes a (subject, relation, object) candidate semantic trace with a
        confidence combining sensor quality and mapping confidence.
        """
        candidates = []
        for raw in raws:
            candidate = kb.map_to_candidate(raw, t)
            if candidate is None:
                continue
            candidates.append(candidate)
            self.local_cognition[candidate.key] = candidate
        return candidates

    def communicate(
        self,
        field: Field,
        observations: list[Observation],
        t: int,
        policy: CommunicationPolicy,
        kb: KnowledgeBase | None = None,
    ) -> list[PolicyDecision]:
        """M1-aware communication: evaluate each candidate semantic trace
        with `policy` (field_sim.communication.CommunicationPolicy - M1Policy
        by default). A candidate the policy declines to transmit is simply
        not written. A candidate that IS transmitted is written to the field
        via Field.set() - a single, unconditional upsert; the field never
        classifies a write as novel/reinforce/refresh/correction.
        """
        decisions = []
        for obs in observations:
            decision = policy.evaluate(obs, field, t, kb)
            if decision.transmit:
                field.set(obs, t)
            decisions.append(decision)
        return decisions

    def receive_field_context(self, field: Field, region_bounds: dict[str, Bounds], t: int) -> M3Result:
        """M3 query: ask field_sim.decision.m3_decide for the area/time/
        confidence-gated contextual subgraph around this agent's own
        current location, plus this agent's own decision-level inferred
        edges - and just return that result.

        Read-only: does NOT write into local_cognition or mutate the agent
        in any way. local_cognition reflects only this agent's own direct
        observation (form_candidate_traces()); there is no merge/overwrite
        path from the field into it at all, deliberately - the caller
        decides what (if anything) to do with the returned M3Result.

        An agent whose own location resolves to no known region
        (resolve_area() returns None - only possible outside
        [0, area_size]^2, or with an empty `region_bounds`; field_sim.
        environment._partition_regions() gaplessly tiles the whole space
        otherwise) gets context_area=None, matching only edges that
        themselves resolve to no area.
        """
        context_area = resolve_area(self.location, region_bounds)
        return m3_decide(field.edges, self.agent_id, context_area, t, region_bounds)


def build_humanoid_linked_agents(
    environment: Environment,
    kb: KnowledgeBase,
    radius: float,
    seed: int,
) -> list[Agent]:
    """No separate sensor population at all - each Humanoid VLA instance IS
    its own sensor: one Agent per Humanoid instance in `environment`,
    sharing that instance's subject name as agent_id (Agent("Humanoid_1",
    ...) observes FOR Humanoid_1) and starting at its current
    field_sim.environment.Environment.locations position. Pedestrian
    instances are never sensors - only Humanoid is (Pedestrian is just a
    walker being observed, not an observer).

    Requires `environment` to have been built with agent_move_step > 0
    (field_sim.environment.Environment) - these Agents don't do their own
    random walk (step_size is irrelevant, only used as a marker here);
    call sync_agent_locations() every round, right after environment.
    step(t), to carry each one to wherever Environment's own drift took the
    matching Humanoid that round. Without agent_move_step > 0, environment.
    _drifting_subjects is empty and this returns NO agents at all; without
    calling sync_agent_locations(), the ones it does return just sit at
    their t=0 position forever.

    Each Humanoid always observes facts about itself regardless of
    observation_radius - see Agent.observe()'s docstring (a per-agent
    guarantee, not something this builder needs to arrange).
    """
    rng = random.Random(seed)
    subjects = [
        s for s in environment._drifting_subjects
        if kb.class_hierarchy.is_a(environment.subject_classes[s], "Humanoid")
    ]
    agents = []
    for i, subject in enumerate(sorted(subjects)):
        agents.append(
            Agent(
                agent_id=subject,  # same id as the VLA instance it's linked to - no separate mapping needed
                location=environment.locations[subject],
                observation_radius=radius,
                step_size=0.0,  # moves only via sync_agent_locations(), never its own random walk
                modality=MODALITIES[i % len(MODALITIES)],
                reliability=rng.uniform(*RELIABILITY_RANGE),
                seed=seed + i,
            )
        )
    return agents


def sync_agent_locations(agents: list[Agent], environment: Environment) -> None:
    """Call once per round, right after environment.step(t), for Agents
    built by build_humanoid_linked_agents - moves each Agent to wherever
    its linked VLA instance's drift took it this round, instead of relying
    on Agent.move()'s own independent random walk (which these Agents
    never use - step_size=0)."""
    for agent in agents:
        if agent.agent_id in environment.locations:
            agent.location = environment.locations[agent.agent_id]
