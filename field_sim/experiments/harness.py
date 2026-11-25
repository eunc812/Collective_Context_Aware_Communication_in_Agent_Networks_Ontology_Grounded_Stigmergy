"""Shared experiment harness for the collective-cognition characterization suite
(experiments CKC / EIR / CCG / persistence).

Purpose: give the four characterization experiments ONE configuration object,
ONE world builder, and ONE structured-results writer, all built on the existing
field_sim modules (Environment, KnowledgeBase, Agent, Field, Simulation) - no new
architecture. Every experiment reuses `build_world(cfg)` so they run over the
same environment / observation / semantic representation / inference rules /
seeds, which is the "keep everything the same across methods" requirement.

Also holds the shared plot color palette (BLUE/ORANGE/AQUA/MUTED/GRID/SURFACE/
INK/RED below) - every figures/make_*.py and exp_*.py script imports these
instead of each keeping its own copy, so figures stay visually consistent.

Fragmentation control: an agent perceives only facts within `observation_radius`
of its location (see Agent.observe). A LARGE radius => each agent sees most of the
environment (LOW fragmentation, redundant coverage); a SMALL radius => each agent
sees a small patch (HIGH fragmentation, knowledge spread across agents). So
`observation_radius` IS the fragmentation knob, no new mechanism needed.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass

from field_sim.agent import build_humanoid_linked_agents
from field_sim.environment import Environment
from field_sim.field import Field
from field_sim.ontology import KnowledgeBase
from field_sim.simulation import Simulation

# Results of the multi-valued-relation model (field_sim.cardinality, triple-
# based coverage - see field_sim.evaluation). The pre-change (single-valued)
# results/figures this superseded are archived in
# backup_pre_multivalued_2026-10-30.zip at the project root, not kept under
# a parallel LEGACY_* directory anymore.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_main")
FIGURES_DIR = os.path.join(_ROOT, "figures")

# Environment change-rate tiers shared by the selective-vs-random grid and
# fresh-vs-stale experiments (moved here unchanged from the removed
# exp_vcr_current): (name, change_prob), one fixed agent_move_step for all.
FIXED_AGENT_MOVE_STEP = 6.0
ENV_RATES = [("slow", 0.02), ("moderate", 0.10), ("fast", 0.35)]

# Shared plot color palette - not every script uses every name.
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
GRID, SURFACE, INK, RED = "#e1e0d9", "#ffffff", "#000000", "#c7401f"

# Shared paper-figure layout (fig_ccg_vs_fragmentation, fig_agent_count_per_agent,
# fig_sync_messages): same figure size and the same axes box in
# inches, saved WITHOUT bbox_inches="tight" so the three PNG/PDFs come out identical in size.
# The margins leave room for a two-line y label and a two-row legend under the axes.
PAPER_FIGSIZE = (6.6, 4.8)                                       # inches
PAPER_AXES_RECT = (1.0 / 6.6, 1.4 / 4.8, 5.35 / 6.6, 3.2 / 4.8)  # left, bottom, width, height (figure fraction)
PAPER_LEGEND_ANCHOR = (0.5, 0.78 / 4.8)                          # figure fraction, loc="upper center"


@dataclass
class ExperimentConfig:
    """All knobs the characterization experiments vary. Defaults are a mild,
    slowly-changing world so coverage/integration effects are not swamped by
    adaptation lag."""
    # None -> field_sim.environment.DEFAULT_INSTANCES (11 instances, this
    # config's long-standing default - exp_roi.py/exp_persist.py both rely
    # on that default and stay unaffected).
    # Pass field_sim.environment.generate_instances(...) explicitly to run
    # over a larger, procedurally-sized VLA population instead - e.g.
    # generate_instances(num_humanoids=8, num_pedestrians=12) - which also
    # scales Area count with agent count so R1/approaching_area and R4/
    # potential_conflict (both require an exact Area, not just any
    # SpatialRegion) don't funnel every conclusion through a single node.
    instances: dict[str, str] | None = None
    # The SENSOR/observer population is no longer independent of `instances`:
    # build_world() builds it via field_sim.agent.build_humanoid_linked_agents,
    # one sensor per Humanoid VLA instance in `instances` (Pedestrian is
    # never a sensor, only ever observed). So the sensor count is controlled
    # by however many Humanoids `instances` declares (DEFAULT_INSTANCES has
    # 2; pass field_sim.environment.generate_instances(num_humanoids=...) for
    # more) - there is no separate num_agents knob any more.
    observation_radius: float = 20.0     # fragmentation knob: large = low fragmentation
    area_size: float = 100.0
    change_prob: float = 0.03            # environment change rate
    evaporation_rate: float = 0.08       # field decay rate
    min_confidence: float = 0.05
    steps: int = 60
    seed: int = 0
    agent_reliability: float | None = None  # if set, forces every agent to this sensor reliability
                                            # (observation noise = 1 - reliability); None = default 0.6-1.0 spread

    # Must be > 0: field_sim.agent.build_humanoid_linked_agents only picks up
    # Humanoid instances from environment._drifting_subjects, which is empty
    # unless agent_move_step > 0 (field_sim.environment.Environment) - a
    # sensor population is a randomly-drifting Humanoid population, there is
    # no other sensor topology any more. See Environment.agent_move_step's
    # docstring for how the drift itself works.
    agent_move_step: float = 6.0


def build_world(cfg: ExperimentConfig, policy_factory=None):
    """Construct a fresh Environment / KnowledgeBase / Agents / Field / Simulation
    from a config. Fresh instances every call (mutable seeded state).

    `policy_factory` (KnowledgeBase -> field_sim.communication.CommunicationPolicy):
    when given, called with THIS call's freshly-built kb (subjects already
    registered) to produce the policy the Simulation is built with - M1
    communication then decides what gets transmitted. It has to be a
    factory rather than a ready policy object: M1Policy is bound to a
    specific KnowledgeBase (its FieldReasoner calls kb.infer()
    against it every evaluate() call), and that has to be THIS kb, not a
    separately-constructed one missing this world's subject registrations.
    None (the default) means every candidate is transmitted unconditionally
    (Simulation's broadcast fallback).
    """
    # Environment now builds its facts FROM a KnowledgeBase's Class/Instance/
    # Property schema (field_sim.environment.Environment), so kb has to
    # exist first; it registers cfg.instances (or DEFAULT_INSTANCES, see
    # ExperimentConfig.instances' docstring) into kb during construction.
    kb = KnowledgeBase()
    environment = Environment(
        kb, instances=cfg.instances, area_size=cfg.area_size, change_prob=cfg.change_prob, seed=cfg.seed,
        agent_move_step=cfg.agent_move_step,
    )
    agents = build_humanoid_linked_agents(environment, kb, radius=cfg.observation_radius, seed=cfg.seed)
    if cfg.agent_reliability is not None:                 # force a uniform sensor-reliability level
        r = max(0.0, min(1.0, cfg.agent_reliability))
        for a in agents:
            a.reliability = r
    field = Field(evaporation_rate=cfg.evaporation_rate, min_confidence=cfg.min_confidence)
    policy = policy_factory(kb) if policy_factory is not None else None
    sim = Simulation(environment, agents, field, kb, policy=policy)
    return environment, kb, agents, field, sim


def write_csv(name: str, rows: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def write_json(name: str, obj) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0
