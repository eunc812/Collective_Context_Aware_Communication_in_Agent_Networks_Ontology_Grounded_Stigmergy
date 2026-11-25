"""Orchestrates the observe -> map -> communicate -> decay -> reason loop.

Each round:
  1. environment evolves (this also drifts every Humanoid, since every
     sensor IS a Humanoid - field_sim.agent.build_humanoid_linked_agents);
     sync_agent_locations() then carries each agent to wherever that drift
     took its linked Humanoid this round.
  2. agents move (a no-op for Humanoid-linked agents, step_size=0.0 - kept
     only for a plain, non-linked Agent population) and perceive facts in
     their own modality (RawObservation)
  3. raw observations are grounded in the shared knowledge base -> semantic
     traces (Observation) via the ontology. Unmappable readings are
     dropped here.
  4. semantic traces are communicated to the field. Two paths:
       - policy is None: every candidate is transmitted unconditionally
         (field_sim.field.Field.set()) - the broadcast ground-truth case.
       - policy is set: M1 (field_sim.communication) - Agent.communicate()
         evaluates every candidate with `policy`; whatever it declines to
         transmit is simply not sent.
  5. the field decays (temporal decay)
  6. the field reasons (M2, field_sim.field.Field.reason()): the knowledge
     base's rules are checked against the field's own edges, and any that
     fire write emergent edges - new cognition emerging from the
     composition of what's already collectively known.
  Agents do NOT read the field back into local cognition at all - local
  cognition reflects only each agent's own direct observation. M3
  (field_sim.decision.m3_decide, via Agent.receive_field_context()) is
  available as an on-demand, read-only query - the area/time/confidence-
  gated context subgraph around an agent's own current location, plus its
  own decision-level inferred edges - but nothing in this loop calls it;
  a caller that wants it (metrics, a future decision-consuming step) calls
  it explicitly against the field after a step.
"""

from dataclasses import dataclass

from field_sim.agent import Agent, sync_agent_locations
from field_sim.communication import CommunicationPolicy
from field_sim.environment import Environment
from field_sim.field import Field
from field_sim.ontology import KnowledgeBase


@dataclass
class StepMetrics:
    t: int
    num_raw_observations: int  # heterogeneous readings the agents perceived this step
    num_candidates: int  # readings successfully grounded by the knowledge base
    num_deposits: int  # semantic traces actually sent to the field this step
    num_field_edges: int
    num_inferences: int  # inference-rule firings this step (emergent edges created/updated)
    accuracy: float  # fraction of ground-truth facts the field currently matches
    avg_confidence: float

    @property
    def num_unmapped(self) -> int:
        """Raw readings the knowledge base couldn't ground (bad vocab, garbled, or constraint violation)."""
        return self.num_raw_observations - self.num_candidates

    @property
    def num_withheld(self) -> int:
        """Candidates formed but M1 declined to transmit (only nonzero when a policy is set)."""
        return self.num_candidates - self.num_deposits


class Simulation:
    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
        field: Field,
        knowledge_base: KnowledgeBase,
        policy: CommunicationPolicy | None = None,
    ):
        self.environment = environment
        self.agents = agents
        self.field = field
        self.knowledge_base = knowledge_base
        # M1 (field_sim.communication): when set, `policy` decides what gets
        # transmitted. None (the default) transmits every candidate
        # unconditionally - the broadcast ground-truth case.
        self.policy = policy

    def step(self, t: int) -> StepMetrics:
        self.environment.step(t)
        sync_agent_locations(self.agents, self.environment)

        num_raw_observations = 0
        num_candidates = 0
        num_deposits = 0
        for agent in self.agents:
            agent.move(self.environment.area_size)
            raws = agent.observe(self.environment, self.knowledge_base, t)
            num_raw_observations += len(raws)

            candidates = agent.form_candidate_traces(raws, self.knowledge_base, t)
            num_candidates += len(candidates)

            if self.policy is not None:
                decisions = agent.communicate(self.field, candidates, t, self.policy, self.knowledge_base)
                num_deposits += sum(1 for d in decisions if d.transmit)
            else:
                for candidate in candidates:
                    self.field.set(candidate, t)
                num_deposits += len(candidates)

        self.field.decay(t)

        inferences = self.field.reason(self.knowledge_base, t)

        return self._metrics(t, num_raw_observations, num_candidates, num_deposits, len(inferences))

    def run(self, steps: int) -> list[StepMetrics]:
        return [self.step(t) for t in range(1, steps + 1)]

    def _metrics(
        self,
        t: int,
        num_raw_observations: int,
        num_candidates: int,
        num_deposits: int,
        num_inferences: int,
    ) -> StepMetrics:
        facts = self.environment.facts
        field_triples = {e.triple for e in self.field.edges.values()}
        matches = sum(1 for fact in facts.values() if fact.triple in field_triples)
        accuracy = matches / len(facts) if facts else 0.0

        confidences = [e.confidence for e in self.field.edges.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return StepMetrics(
            t=t,
            num_raw_observations=num_raw_observations,
            num_candidates=num_candidates,
            num_deposits=num_deposits,
            num_field_edges=len(self.field.edges),
            num_inferences=num_inferences,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
        )
