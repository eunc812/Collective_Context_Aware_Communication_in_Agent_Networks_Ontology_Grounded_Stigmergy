"""Single-axis paper figure for the agent-count experiment
(field_sim.experiments.exp_agent_count): uploads per agent per round versus
the number of agents N = 4, 8, 16, 32, 64, proposed (M1) vs change-triggered,
three change rates (Slow 0.02 / Moderate 0.10 / Quick 0.35).

N = 4 was already present in seed_summary.csv (run in the same batch as
N = 8/16/32, same config.json) but excluded from this figure's original NS
list; it's included here as a full condition, not an afterthought.

  y = total uploads over the run / N / 80 rounds, per seed; line = mean over
      the 8 seeds, shaded band = +-1 SD over the 8 seeds (sample SD)
  x = number of agents N (log2 axis); N = 8 is Case Study 1's population

Based on the saved results only - reads results_additional/agent_count/
seed_summary.csv (N = 4, 8, 16, 32) and seed_summary_n64.csv (N = 64). No
simulation is run.

Saves figures/fig_agent_count_per_agent.png/.pdf (or --out DIR).
Run: python figures/make_agent_count_per_agent.py
"""

import argparse
import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from field_sim.experiments.harness import AQUA, BLUE, FIGURES_DIR, INK, ORANGE, SURFACE
from paper_style import paper_figure, paper_legend, save_paper_figure

DATA_DIR = os.path.join(ROOT, "field_sim", "experiments", "results_additional", "agent_count")
INPUT_CSVS = [os.path.join(DATA_DIR, "seed_summary.csv"), os.path.join(DATA_DIR, "seed_summary_n64.csv")]
NS = [4, 8, 16, 32, 64]
ROUNDS = 80                      # = exp_ccg_vs_fragmentation.STEPS
SEEDS = list(range(8))
RATES = [("Slow", 0.02), ("Moderate", 0.10), ("Quick", 0.35)]
# color = change rate (RATE_COLOR), line style/marker shape are the SAME for both methods
# (solid line, circle marker - matching fig_ccg_vs_fragmentation/fig_sync_messages) - method
# is told apart by fill only: filled circle = Proposed, open/hollow circle = Change-Triggered.
METHODS = [("proposed", "Proposed Method", True), ("change_triggered", "Change-Triggered", False)]
MARKER_SIZE = 7.5
MARKER_EDGE_WIDTH = 1.2
MARKER_SIZE_OPEN = MARKER_SIZE
RATE_COLOR = {"Slow": AQUA, "Moderate": BLUE, "Quick": ORANGE}   # same as the other paper figures
SD_ALPHA = 0.08                  # opacity of the +-1 SD band (same colour as its line)
# display text only - RATES/RATE_COLOR keys stay "Quick" (matches the "rate" column in the saved
# CSVs, written by exp_agent_count.py); only the legend text reads "Fast".
RATE_LABEL = {"Slow": "Slow", "Moderate": "Moderate", "Quick": "Fast"}


def load():
    """-> {(rate, method, N): {seed: uploads per agent per round}}; raises on missing or duplicated seeds."""
    out = {}
    for path in INPUT_CSVS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"input not found: {path} (run exp_agent_count first)")
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n = int(r["N"])
                if n not in NS:
                    continue
                if int(r["agents"]) != n:
                    raise ValueError(f"{path}: N={n} but agents={r['agents']}")
                cell = out.setdefault((r["rate"], r["method"], n), {})
                if int(r["seed"]) in cell:
                    raise ValueError(f"duplicate seed row {r['rate']} {r['method']} N={n} seed={r['seed']}")
                cell[int(r["seed"])] = float(r["uploads"]) / n / ROUNDS
    for label, _ in RATES:
        for m, *_ in METHODS:
            for n in NS:
                if sorted(out.get((label, m, n), {})) != SEEDS:
                    raise ValueError(f"{label} {m} N={n}: seeds {sorted(out.get((label, m, n), {}))}")
    return out


def plot(data, out_dir):
    fig, ax = paper_figure()

    for label, _ in RATES:
        for m, _, filled in METHODS:
            mu = [statistics.mean(data[(label, m, n)].values()) for n in NS]
            sd = [statistics.stdev(data[(label, m, n)].values()) for n in NS]
            color = RATE_COLOR[label]
            ax.fill_between(NS, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)],
                            color=color, alpha=SD_ALPHA, linewidth=0, zorder=2)
            mfc, mec = (color, SURFACE) if filled else (SURFACE, color)
            ms = MARKER_SIZE if filled else MARKER_SIZE_OPEN
            ax.plot(NS, mu, color=color, linestyle="-", linewidth=1.8, marker="o", markersize=ms,
                    markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=MARKER_EDGE_WIDTH, zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlim(NS[0] / 1.15, NS[-1] * 1.15)
    top = max(statistics.mean(data[(label, m, n)].values()) + statistics.stdev(data[(label, m, n)].values())
              for label, _ in RATES for m, *_ in METHODS for n in NS)    # band included
    ax.set_ylim(0, 5 * (int(top // 5) + 1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlabel("Number of Agents", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Uplink Cost per Agent per Round", fontsize=10, color=INK, fontweight="bold")

    # color = change rate, marker fill = method (both solid line + circle marker, as in the
    # other two paper figures). ncol=3 fills column-wise, so this order gives row 1 = the three
    # rates, row 2 = the two methods.
    rate_h = [plt.Line2D([], [], color=RATE_COLOR[l], linewidth=1.8, label=RATE_LABEL[l]) for l, _ in RATES]
    meth_h = [plt.Line2D([], [], color=INK, linestyle="-", linewidth=1.8, marker="o",
                         markersize=(MARKER_SIZE if filled else MARKER_SIZE_OPEN),
                         markerfacecolor=(INK if filled else SURFACE), markeredgecolor=(SURFACE if filled else INK),
                         markeredgewidth=MARKER_EDGE_WIDTH, label=nm) for _, nm, filled in METHODS]
    paper_legend(ax, fig, ncol=3, handles=[rate_h[0], meth_h[0], rate_h[1], meth_h[1], rate_h[2]])
    return save_paper_figure(fig, out_dir, "fig_agent_count_per_agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIGURES_DIR)
    args = ap.parse_args()
    data = load()
    paths = plot(data, args.out)
    for label, _ in RATES:
        for m, nm, *_ in METHODS:
            print(f"{label:<9} {nm:<17}",
                  "  ".join(f"N={n}: {statistics.mean(data[(label, m, n)].values()):5.2f}" for n in NS))
    print("wrote", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
