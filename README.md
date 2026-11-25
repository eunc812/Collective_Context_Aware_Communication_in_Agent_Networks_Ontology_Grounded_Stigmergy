# Ontology-Grounded Stigmergy

Simulation code and experiments for **"Collective Context-Aware Communication for Agent Networks: Ontology-Grounded Stigmergy"**.

The simulation models a population of mobile agents that partially observe a shared environment, selectively transmit semantic observations to a collective context (stigmergic field), and reason over that context with ontology-defined rules. The core mechanism has three parts:

- **M1 — selective transmission**: each agent decides whether to send a candidate observation based on its *state gain* (how novel/stale/confident it is relative to what the collective already holds) and *reasoning gain* (how many new inferences it would unlock).
- **M2 — collective reasoning**: the shared field applies ontology rules (`field_sim/ontology.py`) to its current contents once per round, deriving new triples from what agents have already reported.
- **M3 — decision-level context retrieval**: an agent can query the field for the context relevant to its own current decision.

See `supplementary.tex` (if present) for the full simulation settings and parameter tables.

## Installation

```
pip install -r requirements.txt
```

Python 3.10+ (uses `match`-free modern typing; tested on 3.12). Everything outside of `figures/` is pure standard library; `matplotlib` is only needed to draw the paper figures.

## Repository layout

```
field_sim/                 core simulation library
  agent.py                 partial observation, heterogeneous sensor modalities
  environment.py           world generation, ground-truth facts, movement
  ontology.py              class hierarchy, properties, reasoning rules (M2), grounding
  field.py                 the shared collective context: write/decay/reason
  selection.py             state-gain classification (novel/correction/reinforce/refresh/redundant)
  communication.py         M1 transmission policies (state gain + reasoning gain)
  reasoning.py             fixpoint reasoner used for reasoning-gain evaluation and ground truth
  decision.py               M3: decision-level context retrieval
  evaluation.py             coverage/precision against the reference (ground-truth) state
  provenance.py             traces an inferred triple back to its supporting observations
  change_triggered.py       the "change-triggered" baseline policy (no collective-context check)
  budget_scheduler.py       per-round upload budget / priority scheduling
  conflict_detection.py     potentialConflict event detection for the budget/latency experiment
  cardinality.py            single- vs multi-valued relation storage keys
  types.py                  Observation / FieldEdge / RawObservation dataclasses

field_sim/experiments/     one exp_*.py per reported result; each writes CSV/JSON under
                            results_main/ or results_additional/ (gitignored - see below)
  exp_ccg_vs_fragmentation.py   Case Study 1: collective knowledge gain vs. sensing radius
  exp_agent_count.py            scaling: proposed vs. change-triggered baseline, N = 4..64
  exp_sync_messages.py          periodic delta-synchronization communication cost vs. period K
  exp_decay_sweep.py            decay-rate sensitivity (coverage/precision trade-off)
  exp_alias_mapping.py          heterogeneous-representation / ontology-grounding ablation
  exp_inference_provenance.py   multi-agent support of inferred triples
  exp_budget_conflict_latency.py reasoning gain as upload priority under a fixed budget
  harness.py                     shared config/world-builder/palette used by every script above

figures/                   make_*.py scripts that draw the paper figures from saved results
                            (no simulation is run by these; re-run the matching exp_*.py first);
                            fig_*.pdf are the figures actually used in the paper. fig_*.png
                            (quick-preview renders of the same figures) are gitignored - delete
                            and regenerate freely.
  paper_style.py            shared matplotlib chrome (axes, legend, save) for all figures
```

## Running an experiment

```
python -m field_sim.experiments.exp_ccg_vs_fragmentation
python -m field_sim.experiments.exp_agent_count
python -m field_sim.experiments.exp_sync_messages
# ... etc
```

Each writes its results under `field_sim/experiments/results_main/` or `results_additional/<experiment>/` (`roundwise.csv`, `seed_summary.csv`, `condition_summary.csv`, `summary.json`, `config.json`).

## Regenerating figures

```
python figures/make_ccg_vs_fragmentation.py
python figures/make_agent_count_per_agent.py
python figures/make_sync_messages.py
```

Each reads only the saved CSVs above (no simulation) and writes `figures/fig_*.{png,pdf}`; pass `--out DIR` to write elsewhere.

## Note on saved results

`field_sim/experiments/results_main/` and `results_additional/` are gitignored: they're large (~13 MB combined) and fully regenerable by re-running the `exp_*.py` scripts above with the fixed seeds (0-7) each one uses, which reproduce the exact saved numbers (checked by each script's own `validate()`).
