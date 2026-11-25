"""Single-axis paper figure for the CCG-vs-fragmentation experiment
(field_sim.experiments.exp_ccg_vs_fragmentation): CCG vs upload cost across
observation radii (fragmentation).

  x = upload cost = total deposits per run (log scale, mean over seeds)
  y = CCG = CKC_field - CKC_best_individual (final-window mean over seeds)
  shaded band = +-1 SD of CCG over the 8 seeds (sample SD, vertical only -
      the seed SD of the cost is not drawn), as in fig_agent_count_per_agent
      and fig_sync_messages

Three policies - Proposed Method (M1 + M2), Proposed w/o Collective
Reasoning (same M1 transmission policy, but field.reason()/M2 never runs -
a collective-reasoning ablation), and Proposed w/o Selective Transmission
(broadcast; M2 still runs, but M1's transmission policy is replaced by
sending everything) - one point per observation radius; a thin grey dotted
path joins the proposed and broadcast points for the same radius, and the
radius value is written next to each broadcast point. Styling: harness
palette and paper layout (white background, light grid, legend under the
axes), shared with fig_agent_count_per_agent and fig_sync_messages.

Based on the saved results only - reads results_main/ccg_vs_fragmentation.csv
(means) and the per-seed runs in ccg_vs_fragmentation_summary.json (SD; their
mean is checked against the CSV). No simulation is run.

Saves figures/fig_ccg_vs_fragmentation.png/.pdf (or --out DIR).
Run: python figures/make_ccg_vs_fragmentation.py
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mticker

from field_sim.experiments.harness import AQUA, BLUE, FIGURES_DIR, INK, MUTED, ORANGE, RESULTS_DIR, SURFACE
from paper_style import paper_figure, paper_legend, save_paper_figure

INPUT_CSV = os.path.join(RESULTS_DIR, "ccg_vs_fragmentation.csv")
INPUT_JSON = os.path.join(RESULTS_DIR, "ccg_vs_fragmentation_summary.json")   # per_run: one row per seed
SD_ALPHA = 0.08                  # opacity of the +-1 SD band (as in the other two paper figures)

# policy -> (colour, line style, marker, legend label). Legend entries
# follow this dict's order. Kept simple on purpose: solid line + circle
# marker for every policy, differentiated by color only.
POLICY_STYLE = {
    "selective": (BLUE, "-", "o", "Proposed Method"),
    "selective_no_reasoning": (ORANGE, "-", "o", "w/o Collective Reasoning"),
    "broadcast": (AQUA, "-", "o", "w/o Selective Transmission"),
}


def load():
    """-> {policy: [(radius, cost, ccg)] sorted by radius}; raises on missing/unpaired data."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"input not found: {INPUT_CSV} (run exp_ccg_vs_fragmentation first)")
    data = {p: [] for p in POLICY_STYLE}
    with open(INPUT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] in data:
                data[r["policy"]].append((float(r["radius"]), float(r["cost_deposits"]), float(r["CCG"])))
    for p in data:
        data[p].sort()
    radii = {p: [x[0] for x in data[p]] for p in data}
    policies = list(POLICY_STYLE)
    first = radii[policies[0]]
    if not first or any(radii[p] != first for p in policies[1:]):
        raise ValueError(f"policy radii do not pair: {radii}")
    return data


def load_sd(data):
    """-> {policy: [CCG seed SD] in data[policy]'s radius order}; checks the per-seed mean
    against the CSV mean (rounded to 4 decimals there)."""
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"input not found: {INPUT_JSON} (run exp_ccg_vs_fragmentation first)")
    with open(INPUT_JSON, encoding="utf-8") as f:
        runs = json.load(f)["per_run"]
    sd = {}
    for p, pts in data.items():
        sd[p] = []
        for r, _, ccg in pts:
            vals = [x["ccg"] for x in runs if x["policy"] == p and float(x["radius"]) == r]
            if len(vals) < 2 or abs(statistics.mean(vals) - ccg) > 1e-4:
                raise ValueError(f"{p} r={r}: {len(vals)} seeds, mean {statistics.mean(vals) if vals else None} vs CSV {ccg}")
            sd[p].append(statistics.stdev(vals))
    return sd


def plot(data, sd, out_dir):
    fig, ax = paper_figure()
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)

    # same-radius path: thin grey dotted segments straight from selective to
    # each of the other two policies (selective_no_reasoning, broadcast) -
    # two independent connectors anchored at selective, not a single chain
    # through all three - so the cost gap at that radius between the
    # proposed method and each ablation/baseline stays visible on its own.
    for other in ("selective_no_reasoning", "broadcast"):
        for pts in zip(data["selective"], data[other]):
            xs = [x for _, x, _ in pts]
            ys = [y for _, _, y in pts]
            ax.plot(xs, ys, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.8, zorder=2)
    for policy, (color, ls, marker, label) in POLICY_STYLE.items():
        _, xs, ys = zip(*data[policy])
        # cost rises with radius for every policy, so xs is already increasing
        ax.fill_between(xs, [y - s for y, s in zip(ys, sd[policy])], [y + s for y, s in zip(ys, sd[policy])],
                        color=color, alpha=SD_ALPHA, linewidth=0, zorder=1.5)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8, marker=marker,
                markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=label, zorder=4)
    # three fixed 3pt-by-3pt placements: rising-part radii (r <= peak, except r=10, whose
    # above-left label would land on the steep selective segment) go upper-left; r=10 goes
    # lower-right; falling-part radii (r > peak) go upper-right.
    peak = max(data["broadcast"], key=lambda t: t[2])[0]
    NUDGE = {8.0: (-1, 0), 12.0: (2, 0), 20.0: (0, -1), 26.0: (0, -1), 34.0: (0, -2), 45.0: (0, -2), 60.0: (0, -2)}
    for r, x, y in data["broadcast"]:
        if r == 10.0:
            offset, ha, va = (3, -3), "left", "top"
        elif r <= peak:
            offset, ha, va = (-3, 3), "right", "bottom"
        else:
            offset, ha, va = (3, 3), "left", "bottom"
        dx, dy = NUDGE.get(r, (0, 0))
        offset = (offset[0] + dx, offset[1] + dy)
        ax.annotate(f"r={r:g}", (x, y), textcoords="offset points", xytext=offset, ha=ha, va=va,
                    fontsize=7.5, color=INK, zorder=5)

    # plain-number log ticks
    ax.xaxis.set_major_locator(mticker.FixedLocator([3e3, 5e3, 1e4, 2e4, 5e4]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    allx = [x for p in data for _, x, _ in data[p]]
    ax.set_xlim(min(allx) / 1.02, max(allx) * 1.06)
    # y axis: labelled 0.1 ticks up to the first tenth above the highest point (+ room for the
    # r=26 label), so the top of the data is always read against a labelled tick
    hi = [y + s for p in data for (_, _, y), s in zip(data[p], sd[p])]    # band edges included
    lo = [y - s for p in data for (_, _, y), s in zip(data[p], sd[p])]
    y_top = math.ceil((max(hi) + 0.02) * 10) / 10
    ax.set_ylim(min(-0.02, math.floor((min(lo) - 0.02) * 10) / 10), y_top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Total Uplink Cost", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Collective Knowledge Gain", fontsize=10, color=INK, fontweight="bold")
    paper_legend(ax, fig, ncol=2)
    return save_paper_figure(fig, out_dir, "fig_ccg_vs_fragmentation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIGURES_DIR)
    args = ap.parse_args()
    data = load()
    sd = load_sd(data)
    paths = plot(data, sd, args.out)
    print("wrote", *paths, sep="\n  ")


if __name__ == "__main__":
    main()

