"""Computing the non-dominated set.

The project asks for *the set of non-dominated solutions*, so the method has to
be able to return all of them. Two scalarisation methods are implemented:

`epsilon_constraint`  optimises profit subject to upper bounds on the other two
                      objectives, over a grid of bounds. Because the two
                      constrained objectives are integer valued with small
                      ranges, the grid can be the complete set of integer
                      values, and the method then returns the *exact and
                      complete* Pareto front.

`weighted_sum`        maximises a weighted combination of the three objectives
                      over a grid of weights. Cheaper, but it can only ever
                      return supported solutions - those on the convex hull of
                      the front. It is included to demonstrate that gap, which
                      is the reason the epsilon-constraint method is used for
                      the results.
"""

import itertools
import time

from .model import OBJECTIVES, SENSE, ATTR, solve


# ----------------------------------------------------------------------------
# Dominance
# ----------------------------------------------------------------------------

def _oriented(point):
    """Rewrite a point so that larger is better on every coordinate."""
    return tuple(SENSE[k] * value for k, value in zip(OBJECTIVES, point))


def dominates(a, b):
    """True if point `a` dominates point `b`.

    a dominates b when a is at least as good on every objective and strictly
    better on at least one. Points are (profit, longest project, busiest
    employee); profit is maximised, the other two are minimised.
    """
    pa, pb = _oriented(a), _oriented(b)
    return all(x >= y for x, y in zip(pa, pb)) and any(x > y for x, y in zip(pa, pb))


def pareto_filter(solutions):
    """Keep one solution per non-dominated point, dropping duplicates."""
    kept = []
    for sol in solutions:
        if any(dominates(other.objectives, sol.objectives) for other in solutions):
            continue
        if any(k.objectives == sol.objectives for k in kept):
            continue
        kept.append(sol)
    kept.sort(key=lambda s: s.objectives, reverse=True)
    return kept


# ----------------------------------------------------------------------------
# Payoff table
# ----------------------------------------------------------------------------

def payoff_table(inst, time_limit=None, backend="gurobi"):
    """Optimise each objective in turn, lexicographically.

    Row r of the table is the objective vector obtained by optimising objective
    r first and then tightening the others. The best value in each column is
    the *ideal* point; the worst is the usual estimate of the *nadir* point.
    These give the ranges the epsilon grid has to cover.
    """
    rows = {}
    for first in OBJECTIVES:
        sol = solve(inst, maximise=first, time_limit=time_limit, backend=backend)
        # lexicographic refinement: lock the first objective at its optimum,
        # then optimise the others in turn
        locked = {first: getattr(sol, ATTR[first])}
        for second in OBJECTIVES:
            if second == first:
                continue
            refined = solve(inst, maximise=second, bounds=dict(locked),
                            time_limit=time_limit, backend=backend)
            if refined is not None:
                sol = refined
                locked[second] = getattr(sol, ATTR[second])
        rows[first] = sol.objectives

    ideal, nadir = [], []
    for col, key in enumerate(OBJECTIVES):
        values = [row[col] for row in rows.values()]
        best, worst = (max(values), min(values)) if SENSE[key] > 0 else (min(values), max(values))
        ideal.append(best)
        nadir.append(worst)
    return {"rows": rows, "ideal": tuple(ideal), "nadir": tuple(nadir)}


# ----------------------------------------------------------------------------
# Epsilon-constraint method
# ----------------------------------------------------------------------------

def epsilon_constraint(inst, time_limit=None, duration_max=None, projects_max=None,
                       progress=False, backend="gurobi", step=1):
    """Compute the complete Pareto front by scanning integer epsilon bounds.

    For every pair (e2, e3) the problem solved is

        max  profit        s.t.  longest project <= e2
                                 busiest employee <= e3

    Every Pareto-optimal point is the solution of at least one of these
    problems - it is the one with e2 and e3 set to its own objective values -
    so scanning all integer pairs finds all of them.

    `step` coarsens the grid. step=1 is the exact method described above;
    step=2 scans every second value, which is faster and still returns only
    genuinely non-dominated points, but may no longer return all of them. The
    statement explicitly allows that trade when the exact scan is too costly.

    Most pairs never need a solve. If a looser pair (e2', e3') >= (e2, e3) has
    already been solved and *its* optimal solution happens to respect the
    tighter bounds, then that solution is optimal here too: the tighter problem
    cannot do better than the looser one, and this solution already attains the
    looser optimum. The `reused` counter in the result reports how often this
    applies, which is most of the grid.
    """
    t0 = time.time()
    dur_hi = duration_max if duration_max is not None else inst["h"]
    proj_hi = projects_max if projects_max is not None else len(inst["J"])

    solved = []          # (e2, e3, solution) for every visited cell
    found = []           # solutions actually computed
    n_solves = n_reused = n_unproven = 0

    e2_vals = sorted({*range(dur_hi, -1, -step), 0}, reverse=True)
    e3_vals = sorted({*range(proj_hi, -1, -step), 0}, reverse=True)
    cells = [(e2, e3) for e2 in e2_vals for e3 in e3_vals]
    cells.sort(key=lambda c: (-c[0], -c[1]))     # loosest first

    for e2, e3 in cells:
        # The reuse argument needs the looser cell's answer to be OPTIMAL:
        # "the tighter problem cannot beat the looser one, and this solution
        # already attains the looser optimum". A solve cut off by the time
        # limit attains nothing in particular, so it is never reused.
        reuse = None
        for pe2, pe3, psol in solved:
            if (pe2 >= e2 and pe3 >= e3
                    and (psol.meta or {}).get("optimal")
                    and psol.max_duration <= e2 and psol.max_projects <= e3):
                reuse = psol
                break
        if reuse is not None:
            solved.append((e2, e3, reuse))
            n_reused += 1
            continue

        sol = solve(inst, maximise="profit",
                    bounds={"duration": e2, "projects": e3},
                    time_limit=time_limit, backend=backend)
        n_solves += 1
        if sol is None:                      # cannot happen: doing nothing is feasible
            continue
        solved.append((e2, e3, sol))
        found.append(sol)
        if not (sol.meta or {}).get("optimal"):
            n_unproven += 1
        if progress:
            print(f"  e=({e2:>3},{e3:>3})  ->  {sol.objectives}", flush=True)

    front = pareto_filter(found)
    return {
        "front": front,
        "all": found,
        "n_solves": n_solves,
        "n_reused": n_reused,
        "n_unproven": n_unproven,
        "n_cells": len(cells),
        "runtime": time.time() - t0,
        "grid": (dur_hi, proj_hi),
        "step": step,
        # Exact only when the grid was complete AND every solve proved
        # optimality. Either failing makes this a lower bound on the front.
        "exact": step == 1 and n_unproven == 0,
    }


# ----------------------------------------------------------------------------
# Weighted sum, for comparison only
# ----------------------------------------------------------------------------

def weighted_sum(inst, steps=6, time_limit=None, backend="gurobi"):
    """Maximise w1*profit - w2*duration - w3*projects over a grid of weights.

    Objectives are scaled by their range first, otherwise profit (tens) would
    swamp the other two (single digits) for almost every weight vector.

    This method cannot return unsupported solutions, whatever the weights: a
    weighted sum is a linear function of the objective vector, and its maximum
    over a set of points always lies on the convex hull of that set.
    """
    t0 = time.time()
    table = payoff_table(inst, time_limit=time_limit, backend=backend)
    ideal, nadir = table["ideal"], table["nadir"]
    spans = [max(1, abs(i - n)) for i, n in zip(ideal, nadir)]

    found = []
    grid = [w for w in itertools.product(range(steps + 1), repeat=3)
            if sum(w) == steps and any(w)]
    for w in grid:
        weights = {k: wi / steps / s for k, wi, s in zip(OBJECTIVES, w, spans)}
        sol = _solve_weighted(inst, weights, time_limit, backend)
        if sol is not None:
            found.append(sol)

    return {
        "front": pareto_filter(found),
        "all": found,
        "n_solves": len(grid),
        "runtime": time.time() - t0,
        "weights_tried": len(grid),
    }


def _solve_weighted(inst, weights, time_limit=None, backend="gurobi"):
    """One weighted-sum solve. Kept separate because it bypasses `solve`'s
    single-objective shape."""
    import gurobipy as gp
    from gurobipy import GRB
    from .model import build_model, verify, _run_gurobi, _run_highs

    m, v, f = build_model(inst)
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0        # else a near-optimal point makes the
                               # comparison with epsilon-constraint unfair
    m.setObjective(gp.quicksum(SENSE[k] * weights[k] * f[k] for k in OBJECTIVES),
                   GRB.MAXIMIZE)
    runner = _run_gurobi if backend == "gurobi" else _run_highs
    sol, meta = (runner(inst, m, v) if backend == "gurobi"
                 else runner(inst, m, v, time_limit))
    if sol is None:
        return None
    verify(inst, sol)
    meta.update(weights=weights, backend=backend)
    sol.meta = meta
    return sol


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def front_table(front, inst=None):
    """Readable table of a Pareto front."""
    lines = [f"{'#':>3}  {'profit':>7}  {'longest':>8}  {'per empl.':>10}  {'jobs done':>10}"]
    lines.append("-" * 48)
    for idx, sol in enumerate(front):
        jobs = f"{len(sol.done)}" + (f"/{len(inst['J'])}" if inst else "")
        lines.append(f"{idx:>3}  {sol.profit:>7}  {sol.max_duration:>8}  "
                     f"{sol.max_projects:>10}  {jobs:>10}")
    return "\n".join(lines)
