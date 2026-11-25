"""Ground-truth conflict events and their detection in the collective state.

Events (per relation, default potentialConflict), from the reference closure
of each round's environment (field_sim.evaluation.reference_state):
  - an event of triple x starts at t0 when x is true at t0 and was false at
    t0-1; it lasts while x stays true and ends at the last true round;
  - a later false -> true transition of the same triple is a new event;
  - events already true in the first tracked round are flagged `initial`;
  - events still true in the last round are `censored`.

Detection (checked at the evaluation point of every round, i.e. after
uploads, decay and the live reasoning pass): event x is detected at the
first round td in [t0, end] where
  (1) the field holds x with fact_type "inferred";
  (2) the latest inferred write of x is the stored value and was written at
      or after t0 (field_sim.provenance.ProvenanceRecorder);
  (3) at least one recorded derivation path of that write has ALL of its
      observation leaves still valid: each leaf's observed triple was true in
      the environment in every round from its observation time through td.
latency = td - t0 (0 = detected in the event's first round).

Why (3) excludes stale conclusions: if every leaf of a path had held
continuously since before t0, their conjunction - which is what makes x true
- would already have held before t0, contradicting the event's start. So a
path that passes (3) must rest on at least one observation made during the
current event; a conclusion left over from an earlier event, or merely
rewritten from evidence that has since lapsed, fails it. The final state's
fact_type alone is never used.
"""

from field_sim.cardinality import edge_key

RELATION = "potentialConflict"


class ConflictTracker:
    def __init__(self, relation: str = RELATION):
        self.relation = relation
        self.events: list[dict] = []
        self.active: dict[tuple, dict] = {}
        self.env_history: dict[int, set] = {}
        self.first_round: int | None = None
        self.last_round: int | None = None
        self.stale_leaf_violations = 0       # sanity: valid paths whose leaves all predate t0 (should be 0)

    def on_round(self, t: int, truth: set, env_triples: set, field, recorder) -> None:
        """truth: reference-closure triples at round t; env_triples: the
        environment's observable facts at round t (as triples)."""
        if self.first_round is None:
            self.first_round = t
        self.last_round = t
        self.env_history[t] = set(env_triples)
        now = {x for x in truth if x[1] == self.relation}
        for x, ev in list(self.active.items()):
            if x not in now:
                ev["t_end"] = t - 1
                del self.active[x]
        for x in sorted(now - set(self.active)):
            ev = {"triple": x, "t0": t, "t_end": None, "initial": t == self.first_round, "censored": False,
                  "td": None}
            self.events.append(ev)
            self.active[x] = ev
        for x, ev in self.active.items():
            if ev["td"] is None and self.detected(x, ev["t0"], t, field, recorder):
                ev["td"] = t

    def finalize(self) -> list[dict]:
        for ev in self.active.values():
            ev["t_end"] = self.last_round
            ev["censored"] = True
        self.active = {}
        for ev in self.events:
            ev["duration"] = ev["t_end"] - ev["t0"] + 1
            ev["detected"] = ev["td"] is not None
            ev["latency"] = ev["td"] - ev["t0"] if ev["detected"] else None
            ev["missed"] = not ev["detected"]
            ev["missed_censored"] = ev["missed"] and ev["censored"]
        return self.events

    def detected(self, x: tuple, t0: int, t: int, field, recorder) -> bool:
        edge = field.edges.get(edge_key(*x))
        if edge is None or edge.triple != x or edge.fact_type != "inferred":
            return False
        vid = recorder.latest_inferred.get(edge.key)
        if vid is None:
            return False
        v = recorder.versions[vid]
        if v.triple != x or v.t < edge.value_since or v.t < t0:
            return False
        paths, _unresolved = recorder.leaf_version_paths(vid)
        for leaves in paths:
            if all(self._valid_through(recorder.versions[lv], t) for lv in leaves):
                if all(recorder.versions[lv].obs_time < t0 for lv in leaves):
                    self.stale_leaf_violations += 1
                return True
        return False

    def _valid_through(self, leaf, t: int) -> bool:
        return all(leaf.triple in self.env_history.get(r, ()) for r in range(leaf.obs_time, t + 1))
