"""Inference provenance for the shared field - an ANALYSIS-ONLY recorder.

FieldEdge's own provenance fields cannot answer "did this inferred triple
come from observations of two or more agents along one actual derivation":
  - `sources` is the union of every agent that ever wrote the key, and of
    every premise's sources over every firing the edge ever had - repeated
    reports of one fact, old values and independent derivation paths are
    all merged into one set;
  - `rule_id` / `condition_sources` cover only the most recent pass, merged
    across all of its firings, and name premise KEYS, not the premise
    versions (object, writer, time) that were actually used.

A ProvenanceRecorder attached as `Field.provenance` logs every Field.set()
as an immutable *version* and, for inferred writes, the firings
(rule, premise version ids) Field.reason() used - premise versions taken
from the reasoning snapshot, i.e. as they stood when the inference was made.
The recorder only reads what Field already computed; nothing in the
simulation reads the recorder, so attaching it cannot change any decision.

Classification of a currently stored inferred edge (classify()):
  - its derivation is the most recent inferred write of that key (the one
    that produced the stored object); each firing of that write is a
    separate path, and an inferred premise expands into each of its own
    recorded paths - paths are never merged with each other;
  - a path's leaves are the observed-write versions it bottoms out in, as
    (triple, reporting agent) pairs;
  - a path is MULTI-AGENT iff two of its leaves are different triples
    reported by different agents. One triple reported by several agents is
    not enough on its own;
  - the edge is "multi" if at least one resolved path is multi-agent,
    "single" if every path resolved and none is, and "unresolved" otherwise
    (a premise version missing from the log, or a path count above
    path_cap). Because a premise version always predates the version built
    from it, expansion follows strictly decreasing version ids and always
    terminates, including through near_symmetric.
"""

from dataclasses import dataclass
from itertools import product

PATH_CAP = 4096
MULTI, SINGLE, UNRESOLVED = "multi", "single", "unresolved"


@dataclass(frozen=True)
class Version:
    vid: int
    key: tuple
    triple: tuple
    t: int
    kind: str                   # "observed" | "inferred" (the fact_type of this write)
    agent: str | None           # observed: the reporting agent
    obs_time: int | None
    firings: tuple | None       # inferred: ((rule, (premise vid | None, ...)), ...)


@dataclass(frozen=True)
class Classification:
    status: str                 # MULTI | SINGLE | UNRESOLVED
    also_single: bool           # MULTI and a single-agent path also exists
    vid: int | None
    reason: str = ""


def is_multi_agent(leaves: frozenset) -> bool:
    """Two leaves that differ in BOTH the observed triple and the reporting agent."""
    leaves = list(leaves)
    return any(t1 != t2 and a1 != a2 for i, (t1, a1) in enumerate(leaves) for t2, a2 in leaves[i + 1:])


class ProvenanceRecorder:
    def __init__(self, path_cap: int = PATH_CAP):
        self.path_cap = path_cap
        self.versions: list[Version] = []
        self.current: dict[tuple, int] = {}            # key -> latest version id
        self.latest_inferred: dict[tuple, int] = {}    # key -> latest inferred-write version id
        self._pending: dict[tuple, tuple] = {}
        self._memo: dict[int, tuple[frozenset, bool]] = {}
        self._leaf_memo: dict[int, tuple[frozenset, bool]] = {}

    # --- hooks called by Field -------------------------------------------
    def begin_inference(self, groups: dict[tuple, list[tuple[str, tuple]]]) -> None:
        """Called by Field.reason() before its first write: {conclusion key:
        [(rule, premise keys), ...]} for every firing about to be written.
        Premise keys resolve to their versions now, at the snapshot."""
        self._pending = {
            key: tuple((rule, tuple(self.current.get(k) for k in keys)) for rule, keys in firings)
            for key, firings in groups.items()
        }

    def on_set(self, obs, t: int, fact_type: str) -> None:
        vid = len(self.versions)
        key = obs.key
        if fact_type == "inferred":
            v = Version(vid, key, obs.triple, t, "inferred", None, obs.obs_time, self._pending.pop(key, None))
            self.latest_inferred[key] = vid
        else:
            v = Version(vid, key, obs.triple, t, "observed", obs.agent_id, obs.obs_time, None)
        self.versions.append(v)
        self.current[key] = vid

    # --- analysis ----------------------------------------------------------
    def paths(self, vid: int) -> tuple[frozenset, bool]:
        """(set of resolved paths as frozensets of (triple, agent) leaves,
        whether any path could not be resolved)."""
        if vid in self._memo:
            return self._memo[vid]
        v = self.versions[vid]
        if v.kind == "observed":
            out = (frozenset({frozenset({(v.triple, v.agent)})}), False)
        elif v.firings is None:
            out = (frozenset(), True)
        else:
            found, unresolved = set(), False
            for _rule, premise_vids in v.firings:
                if any(p is None or p >= vid for p in premise_vids):
                    unresolved = True
                    continue
                children = [self.paths(p) for p in premise_vids]
                unresolved |= any(u for _, u in children)
                for combo in product(*(c for c, _ in children)):
                    found.add(frozenset().union(*combo))
                    if len(found) > self.path_cap:
                        unresolved = True
                        break
            out = (frozenset(found), unresolved)
        self._memo[vid] = out
        return out

    def leaf_version_paths(self, vid: int) -> tuple[frozenset, bool]:
        """Like paths(), but each path is the frozenset of its observed-write
        VERSION ids, so a caller can check each leaf's observation time and
        content (e.g. field_sim.conflict_detection)."""
        if vid in self._leaf_memo:
            return self._leaf_memo[vid]
        v = self.versions[vid]
        if v.kind == "observed":
            out = (frozenset({frozenset({vid})}), False)
        elif v.firings is None:
            out = (frozenset(), True)
        else:
            found, unresolved = set(), False
            for _rule, premise_vids in v.firings:
                if any(p is None or p >= vid for p in premise_vids):
                    unresolved = True
                    continue
                children = [self.leaf_version_paths(p) for p in premise_vids]
                unresolved |= any(u for _, u in children)
                for combo in product(*(c for c, _ in children)):
                    found.add(frozenset().union(*combo))
                    if len(found) > self.path_cap:
                        unresolved = True
                        break
            out = (frozenset(found), unresolved)
        self._leaf_memo[vid] = out
        return out

    def classify(self, edge) -> Classification:
        vid = self.latest_inferred.get(edge.key)
        if vid is None:
            return Classification(UNRESOLVED, False, None, "no inferred write logged")
        v = self.versions[vid]
        if v.triple != edge.triple or v.t < edge.value_since:
            return Classification(UNRESOLVED, False, vid, "latest inferred write is not the stored value")
        found, unresolved = self.paths(vid)
        multi = any(is_multi_agent(p) for p in found)
        single = any(not is_multi_agent(p) for p in found)
        if multi:
            return Classification(MULTI, single, vid)
        if unresolved or not found:
            return Classification(UNRESOLVED, False, vid, "path not traceable to observations")
        return Classification(SINGLE, False, vid)

    def tree(self, vid: int, depth: int = 0, max_depth: int = 8) -> dict:
        """Readable derivation tree of one version (every recorded firing)."""
        v = self.versions[vid]
        node = {"triple": list(v.triple), "kind": v.kind, "written_round": v.t}
        if v.kind == "observed":
            node.update(agent=v.agent, obs_time=v.obs_time)
        elif v.firings is not None and depth < max_depth:
            node["firings"] = [
                {"rule": rule, "premises": [self.tree(p, depth + 1, max_depth) if p is not None else None
                                            for p in pv]}
                for rule, pv in v.firings
            ]
        return node
