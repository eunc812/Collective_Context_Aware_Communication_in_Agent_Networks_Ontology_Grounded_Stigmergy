"""Shared paper-figure chrome for figures/make_*.py.

The axes/background/spines/grid/legend/save boilerplate that fig_ccg_vs_
fragmentation, fig_sync_messages and fig_agent_count_per_agent each used to
repeat near-identically (copied by hand, so they only stayed in sync as long
as every edit was copied to all three) is factored out here once, so every
paper figure built from these three helpers is visually identical by
construction instead of by manual upkeep.

Deliberately NOT part of field_sim.experiments.harness: harness.py is also
imported by the non-plotting exp_*.py scripts, some of which run inside
ProcessPoolExecutor workers - keeping matplotlib out of that import path
avoids paying its (non-trivial) import cost in every worker process that
never plots anything. Only figures/make_*.py scripts import this module,
and they already import matplotlib themselves before doing so.
"""

import os

import matplotlib.pyplot as plt

from field_sim.experiments.harness import GRID, INK, PAPER_AXES_RECT, PAPER_FIGSIZE, PAPER_LEGEND_ANCHOR, SURFACE

YLABEL_COORDS = (-0.11, 0.5)     # y-label position in axes coordinates (x, y), same in all figures


def paper_figure():
    """-> (fig, ax) with the shared paper-figure size/axes box, background,
    spines, grid and tick styling every fig_*.py in this set uses."""
    fig = plt.figure(figsize=PAPER_FIGSIZE)
    ax = fig.add_axes(PAPER_AXES_RECT)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # pin the y-label at the same spot in every figure: by default matplotlib puts it just
    # left of the tick labels, so its position (and the visible left margin) shifted with the
    # tick-label width ("0.8" / "30" / "25,000"). Fixed far enough left for the widest one.
    ax.yaxis.set_label_coords(*YLABEL_COORDS)
    return fig, ax


def paper_legend(ax, fig, ncol, handles=None, labels=None):
    """Shared legend styling: positioned under the axes at
    PAPER_LEGEND_ANCHOR (figure-fraction coordinates), bold text, SURFACE/
    GRID frame. Pass `handles` (and optionally `labels`) for a figure that
    builds its own proxy handles (e.g. fig_agent_count_per_agent's rate x
    method combination); omit both to legend whatever the plotted artists
    were already given a `label=` for."""
    kwargs = {}
    if handles is not None:
        kwargs["handles"] = handles
    if labels is not None:
        kwargs["labels"] = labels
    leg = ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=PAPER_LEGEND_ANCHOR,
                     bbox_transform=fig.transFigure, ncol=ncol, frameon=True, handlelength=2.5,
                     columnspacing=1.2, **kwargs)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    return leg


def save_paper_figure(fig, out_dir, fig_name):
    """Save {fig_name}.png/.pdf into out_dir at the shared dpi/facecolor, no
    tight bbox - every paper figure keeps the exact same PAPER_FIGSIZE so
    the three come out identically sized."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{fig_name}.{ext}") for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return paths
