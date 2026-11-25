"""Change-triggered transmission: the context-free baseline.

Each agent keeps ONLY what it itself last sent, per storage key
(field_sim.cardinality.edge_key - (subject, relation) for a single-valued
relation, the full triple for a multi-valued one). A candidate is sent iff
the agent has never sent that key, or the object differs from the object it
last sent for that key. Nothing else is consulted: no shared/collective
state, no other agent's sends, no confidence, no time. The decision is a
pure function of this agent's own send history and the candidate.
"""


class ChangeTriggeredPolicy:
    def __init__(self):
        self.last_sent: dict[str, dict[tuple, str]] = {}

    def should_send(self, obs) -> bool:
        mine = self.last_sent.get(obs.agent_id, {})
        return mine.get(obs.key) != obs.object if obs.key in mine else True

    def record_sent(self, obs) -> None:
        self.last_sent.setdefault(obs.agent_id, {})[obs.key] = obs.object
