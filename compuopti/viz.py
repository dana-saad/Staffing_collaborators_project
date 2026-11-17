"""Plots for both spaces the project asks about.

Decision space  `plot_schedule` - who works on what, day by day.
Objective space `plot_front_3d`, `plot_front_projections`, `plot_parallel` -
                where the non-dominated solutions sit relative to each other.

Colour
------
Categorical hues come from a fixed, colour-vision-deficiency-checked order and
are assigned in that order, never cycled, wherever the number of series is
small enough (the objective-space plots).

`plot_schedule` is the one deliberate exception. A schedule can carry 25 jobs,
far past any safe categorical palette, so there colour is a grouping aid only:
identity is carried by the text written inside every cell, which names the job
and the qualification. No reader ever has to tell two fills apart to know what
a cell is.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from .model import LABELS

#: fixed categorical order, validated for colour-vision deficiency
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
LEAVE = "#e8e7e3"


def _clean(ax, grid_axis="both"):
    """Recessive axes: no box, soft grid behind the data."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_SOFT, labelsize=9)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK_SOFT)
        label.set_fontsize(10)


# ----------------------------------------------------------------------------
# Decision space
# ----------------------------------------------------------------------------

def plot_schedule(inst, sol, title=None, ax=None, cell_text=True):
    """One row per person, one column per day; a filled cell is an assignment.

    Each cell is labelled with the job number and the qualification worked, so
    the schedule can be read without reference to the legend.
    """
    S, H, J = inst["S"], inst["H"], inst["J"]
    colour = {j: SERIES[idx % len(SERIES)] for idx, j in enumerate(J)}

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 0.46 * len(H) + 3.2),
                                      0.52 * len(S) + 1.9))

    for row, person in enumerate(S):
        for t in H:
            x, y = t - 1, row
            if t in inst["V"][person]:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=LEAVE,
                                           edgecolor="white", linewidth=1.4))
            elif (person, t) in sol.schedule:
                job, qual = sol.schedule[person, t]
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=colour[job],
                                           edgecolor="white", linewidth=1.4))
                if cell_text:
                    ax.text(x + 0.5, y + 0.5, f"{job.replace('Job', '')}\n{qual}",
                            ha="center", va="center", fontsize=7.0,
                            color="white", linespacing=0.95)
            else:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="white",
                                           edgecolor=GRID, linewidth=0.7))

    ax.set_xlim(0, len(H))
    ax.set_ylim(0, len(S))
    ax.invert_yaxis()
    ax.set_xticks([t - 0.5 for t in H])
    ax.set_xticklabels(H, fontsize=8)
    ax.set_yticks([r + 0.5 for r in range(len(S))])
    ax.set_yticklabels(S, fontsize=9)
    ax.tick_params(length=0, colors=INK_SOFT)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("day", color=INK_SOFT, fontsize=10)

    handles = []
    for job in J:
        if job in sol.done:
            late = sol.late.get(job, 0)
            label = job + (f"  +{late}d late" if late else "")
            handles.append(Patch(facecolor=colour[job], label=label))
    handles.append(Patch(facecolor=LEAVE, label="on leave"))
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              fontsize=8, frameon=False, labelcolor=INK_SOFT,
              ncol=1 if len(handles) <= 14 else 2)

    if title is None:
        title = (f"profit {sol.profit}   longest project {sol.max_duration}d   "
                 f"max {sol.max_projects} projects per person")
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    return ax


# ----------------------------------------------------------------------------
# Objective space
# ----------------------------------------------------------------------------

def plot_front_3d(front, title="Non-dominated solutions", ax=None, annotate=3):
    """The front in the three objectives at once.

    Colour encodes profit - one hue, light to dark, because profit is a
    magnitude - so the third dimension stays readable when the figure is
    printed or seen at an angle.
    """
    points = np.array([s.objectives for s in front], dtype=float)
    if ax is None:
        fig = plt.figure(figsize=(7.2, 5.6))
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 1], points[:, 2], points[:, 0],
               c=points[:, 0], cmap="Blues", vmin=points[:, 0].min() - 10,
               s=70, edgecolor=INK_SOFT, linewidth=0.6, depthshade=False)

    # label the extreme points only - never a number on every mark
    order = np.argsort(-points[:, 0])
    for idx in order[:annotate]:
        ax.text(points[idx, 1], points[idx, 2], points[idx, 0],
                f"  {int(points[idx, 0])}", fontsize=8, color=INK_SOFT)

    ax.set_xlabel(LABELS["duration"], fontsize=9, color=INK_SOFT, labelpad=8)
    ax.set_ylabel(LABELS["projects"], fontsize=9, color=INK_SOFT, labelpad=8)
    ax.set_zlabel(LABELS["profit"], fontsize=9, color=INK_SOFT, labelpad=6)
    ax.tick_params(labelsize=8, colors=INK_SOFT)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(integer=True))
    ax.view_init(elev=20, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor(GRID)
    return ax


def plot_front_projections(front, axes=None, title=None):
    """The three pairwise views of the front.

    In each panel the remaining objective is shown by marker size, so a point
    that looks dominated in two dimensions can be seen to survive on the third.
    """
    points = np.array([s.objectives for s in front], dtype=float)
    pairs = [(1, 0, 2), (2, 0, 1), (1, 2, 0)]
    keys = ["profit", "duration", "projects"]

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for ax, (xi, yi, si) in zip(np.ravel(axes), pairs):
        size = points[:, si]
        span = size.max() - size.min()
        scaled = 40 + 170 * ((size - size.min()) / span if span else np.zeros_like(size))
        ax.scatter(points[:, xi], points[:, yi], s=scaled, c=SERIES[0],
                   alpha=0.75, edgecolor="white", linewidth=1.4, zorder=3)
        ax.set_xlabel(LABELS[keys[xi]])
        ax.set_ylabel(LABELS[keys[yi]])
        ax.set_title(f"marker size: {LABELS[keys[si]]}", loc="left",
                     fontsize=9, color=INK_SOFT)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _clean(ax)

    if title:
        plt.gcf().suptitle(title, x=0.012, ha="left", color=INK, fontsize=12)
        plt.gcf().tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.gcf().tight_layout()
    return axes


def plot_parallel(front, ax=None, title="Every non-dominated solution",
                  highlight=None):
    """Parallel coordinates: one line per solution across the three objectives.

    Each axis is drawn so that *up is better*, which turns the trade-off into
    something you can see directly: no line can be above every other line
    everywhere, or the solutions it beat would not be on the front.
    """
    points = np.array([s.objectives for s in front], dtype=float)
    keys = ["profit", "duration", "projects"]
    # normalise each column to [0, 1] with 1 = best
    norm = np.zeros_like(points)
    for col, key in enumerate(keys):
        lo, hi = points[:, col].min(), points[:, col].max()
        span = hi - lo
        scaled = (points[:, col] - lo) / span if span else np.full(len(points), 0.5)
        norm[:, col] = scaled if key == "profit" else 1 - scaled

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.6))

    xs = np.arange(3)
    for row, raw in zip(norm, points):
        is_hl = highlight is not None and tuple(int(x) for x in raw) == tuple(highlight)
        ax.plot(xs, row, color=SERIES[1] if is_hl else SERIES[0],
                linewidth=2.4 if is_hl else 1.6,
                alpha=1.0 if is_hl else 0.45, zorder=4 if is_hl else 2,
                marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.2)

    for col, key in enumerate(keys):
        lo, hi = points[:, col].min(), points[:, col].max()
        best, worst = (hi, lo) if key == "profit" else (lo, hi)
        ax.annotate(f"{int(best)}", (col, 1.0), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=INK_SOFT)
        ax.annotate(f"{int(worst)}", (col, 0.0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=INK_SOFT)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.12)
    ax.set_ylabel("worse  →  better", fontsize=9)
    _clean(ax, grid_axis="x")
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=12)
    return ax


def plot_method_comparison(eps_front, ws_front, ax=None):
    """Epsilon-constraint front against the weighted-sum front.

    The points the weighted sum misses are the unsupported ones: non-dominated
    solutions that lie inside the convex hull of the front, where no set of
    weights can make them the unique maximum of a linear objective.
    """
    eps = np.array([s.objectives for s in eps_front], dtype=float)
    ws = {s.objectives for s in ws_front}
    missed = np.array([p for p in eps if tuple(int(x) for x in p) not in ws], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.scatter(eps[:, 1], eps[:, 0], s=120, facecolor="none",
               edgecolor=SERIES[0], linewidth=1.8,
               label=f"epsilon-constraint ({len(eps_front)} points)", zorder=3)
    if len(missed):
        ax.scatter(missed[:, 1], missed[:, 0], s=52, color=SERIES[1],
                   edgecolor="white", linewidth=1.2,
                   label=f"missed by weighted sum ({len(missed)})", zorder=4)

    ax.set_xlabel(LABELS["duration"])
    ax.set_ylabel(LABELS["profit"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _clean(ax)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SOFT, loc="lower right")
    ax.set_title("Why the weighted sum is not enough", loc="left",
                 color=INK, fontsize=11, pad=10)
    return ax


# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------

def plot_scaling(rows, x="n_vars", ax=None):
    """Solve time against instance size, on a log y axis.

    `rows` is the list of dicts produced by the performance study.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))

    xs = [r[x] for r in rows]
    for idx, (key, label) in enumerate([("t_single", "one profit-optimal solve"),
                                        ("t_front", "complete Pareto front")]):
        ys = [r[key] for r in rows if r.get(key) is not None]
        xs_k = [r[x] for r in rows if r.get(key) is not None]
        ax.plot(xs_k, ys, marker="o", markersize=7, linewidth=2,
                color=SERIES[idx], markeredgecolor="white", markeredgewidth=1.2,
                label=label)
        if ys:
            ax.annotate(f"{ys[-1]:.0f}s", (xs_k[-1], ys[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=8, color=INK_SOFT, va="center")

    ax.set_yscale("log")
    ax.set_xlabel("variables in the model")
    ax.set_ylabel("seconds")
    _clean(ax)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SOFT, loc="upper left")
    ax.set_title("How the method scales", loc="left", color=INK, fontsize=11, pad=10)
    return ax
