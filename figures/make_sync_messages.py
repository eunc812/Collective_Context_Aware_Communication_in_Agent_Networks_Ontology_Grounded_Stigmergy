"""Single-axis paper figure for the synchronization-period experiment
(field_sim.experiments.exp_sync_messages): communication cost in messages
versus the synchronization period K = 1..20, at ONE environment: the slow
change rate (p = 0.02), Case Study 1's default (exp_ccg_vs_fragmentation.
CHANGE_PROB). The other two rates (0.10 / 0.35) are in the saved results but
not drawn.

  uplink    uploaded traces (all 8 agents), 80-round total
  downlink  delta entries broadcast by the edge (once per broadcast)
  total     uplink + downlink
The free-access reference (K = 0, = Case Study 1) is not drawn; its cost
is printed.

Line = mean over the 8 seeds; shaded band = +-1 SD over the 8 seeds (sample
SD), as in fig_agent_count_per_agent. The per-rate values are in
condition_summary.csv.

Based on the saved results only - reads results_additional/sync_messages/
seed_summary.csv. No simulation is run.

Saves figures/fig_sync_messages.png/.pdf (or --out DIR).
Run: python figures/make_sync_messages.py
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
import matplotlib.ticker as mticker

from field_sim.experiments.harness import AQUA, BLUE, FIGURES_DIR, INK, ORANGE, SURFACE
from paper_style import paper_figure, paper_legend, save_paper_figure

INPUT_CSV = os.path.join(ROOT, "field_sim", "experiments", "results_additional", "sync_messages", "seed_summary.csv")
KS = list(range(1, 21))
SEEDS = list(range(8))
RATES = ["Slow", "Moderate", "Quick"]   # all checked present in load(); only RATE is drawn
RATE = "Slow"                           # p = 0.02, Case Study 1's change rate
SD_ALPHA = 0.08                         # opacity of the +-1 SD band (as in fig_agent_count_per_agent)
# metric -> legend label, colour, line style, marker: same colour scheme as
# fig_ccg_vs_fragmentation (blue/aqua/orange), all solid lines, circle marker
# for every series - kept simple; colour alone tells the series apart. Legend
# order = this list's order.
SERIES = [("total", "Total", BLUE, "-", "o"), ("uplink", "Uplink", AQUA, "-", "o"),
          ("downlink", "Downlink", ORANGE, "-", "o")]


def load():
    """-> {(rate, K, seed): {metric: value}}; raises on missing or duplicated rows."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"input not found: {INPUT_CSV} (run exp_sync_messages first)")
    out = {}
    with open(INPUT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["rate"], int(r["K"]), int(r["seed"]))
            if key in out:
                raise ValueError(f"duplicate row {key}")
            out[key] = {m: float(r[m]) for m, *_ in SERIES}
    missing = [(r, k, s) for r in RATES for k in [0] + KS for s in SEEDS if (r, k, s) not in out]
    if missing:
        raise ValueError(f"{len(missing)} rows missing, e.g. {missing[0]}")
    return out


def seeds_of(data, k, metric):
    """Values over the 8 seeds at the drawn change rate RATE."""
    return [data[(RATE, k, s)][metric] for s in SEEDS]


def plot(data, out_dir):
    fig, ax = paper_figure()

    top = 0.0
    for m, name, color, ls, marker in SERIES:
        mu = [statistics.mean(seeds_of(data, k, m)) for k in KS]
        sd = [statistics.stdev(seeds_of(data, k, m)) for k in KS]
        ax.fill_between(KS, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)],
                        color=color, alpha=SD_ALPHA, linewidth=0, zorder=2)
        ax.plot(KS, mu, color=color, linestyle=ls, linewidth=1.8, marker=marker, markersize=7.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=4)
        top = max(top, max(a + b for a, b in zip(mu, sd)))

    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    # labelled 5,000 ticks up to the first one above the highest band edge, as in figure 1
    ax.set_ylim(0, 5000 * (int(top // 5000) + 1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5000))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Synchronization Period", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Communication Cost", fontsize=10, color=INK, fontweight="bold")
    paper_legend(ax, fig, ncol=3)
    return save_paper_figure(fig, out_dir, "fig_sync_messages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIGURES_DIR)
    args = ap.parse_args()
    data = load()
    paths = plot(data, args.out)
    print(f"{RATE}: free access (K=0) total {statistics.mean(seeds_of(data, 0, 'total')):.0f}")
    for k in KS:
        print(f"K={k:2d}", "  ".join(f"{m} {statistics.mean(seeds_of(data, k, m)):7.0f}"
                                     f"±{statistics.stdev(seeds_of(data, k, m)):5.0f}" for m, *_ in SERIES))
    print("wrote", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
