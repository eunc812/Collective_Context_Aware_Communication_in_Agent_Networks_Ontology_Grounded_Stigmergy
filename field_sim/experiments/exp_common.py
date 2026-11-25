"""Shared helpers for exp_decay_sweep and exp_alias_mapping (moved verbatim
from the removed exp_sync_cost module; behaviour unchanged).

Case Study 1 configuration at r = 26 with a chosen environment change
probability, unique-triple quality, mean/SD over seeds, CSV writing, and a
content manifest used to check that preserved result folders are untouched.
"""

import csv
import hashlib
import math
import os
import statistics

from field_sim.experiments import exp_ccg_vs_fragmentation as ccg

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RADIUS = 26.0
SEEDS = list(range(8))
RATES = {0.02: "Slow", 0.10: "Moderate", 0.35: "Quick"}
PRESERVED = [os.path.join(os.path.dirname(__file__), d) for d in
             ("results_main", os.path.join("results_additional", "provenance"),
              os.path.join("results_additional", "budget_conflict"))]


def quality(held: set, truth: set) -> dict:
    matched = len(held & truth)
    if not truth:
        raise ValueError("empty truth - coverage undefined")
    return {"coverage": matched / len(truth), "precision": matched / len(held) if held else math.nan,
            "matched": matched, "held": len(held), "truth_size": len(truth)}


def _cfg(rate, seed, evaporation=None):
    cfg = ccg._cfg(RADIUS, seed)
    cfg.change_prob = rate
    if evaporation is not None:
        cfg.evaporation_rate = evaporation
    return cfg


def _ms(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (math.nan, math.nan)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(rows)


def manifest(paths):
    out = {}
    for root in paths:
        for dp, _, fn in os.walk(root):
            for f in fn:
                p = os.path.join(dp, f)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, _ROOT)] = hashlib.sha256(fh.read()).hexdigest()
    return out
