"""Independent checks on a computed Pareto front.

`epsilon_constraint` is correct by construction, but "correct by construction"
is exactly the kind of claim that should be tested rather than asserted, so
this module re-derives the same conclusions a different way.

Completeness is an argument rather than a test, and it is worth stating
precisely because it is the thing a jury is most likely to ask about:

    Let P = (p, d, q) be any Pareto-optimal point. Since d <= h and q <= |J|,
    the cell (e2, e3) = (d, q) is in the scanned grid. At that cell the method
    maximises profit subject to duration <= d and projects <= q. P itself is
    feasible there, so the optimum p* satisfies p* >= p. If p* > p the optimal
    solution of that cell would dominate P, contradicting P being Pareto. So
    p* = p, and the augmentation term makes the solver return the solution with
    the smallest duration and projects among those reaching profit p, which is
    P. Hence P is found. The scan therefore misses no Pareto-optimal point.

What is tested here is the other direction - that every point reported is
genuinely non-dominated - plus a randomised search for any point the front
might have missed.
"""

import random

from .model import solve, verify
from .pareto import dominates


def check_point_nondominated(inst, sol, time_limit=None, backend="gurobi"):
    """Look for a feasible solution that dominates `sol`, with three solves.

    A dominating point would have to be better on at least one objective while
    no worse on the others. Each of the three solves asks whether that is
    possible for one objective, holding the other two at `sol`'s values.

    Returns (ok, detail). `ok` is False only when a solve actually *found*
    something strictly better, which proves domination whether or not that
    solve converged - any incumbent is a feasible solution. `detail` carries
    a "proved" flag saying whether all three solves reached optimality; if it
    is False, `ok=True` means "no dominating solution found within the time
    limit" rather than "proved non-dominated".

    Reading a non-converged solve as if it were optimal is the trap here. A
    time-limited minimisation can return a value *worse* than the point's own,
    which looks like a contradiction and is really just an unfinished search.
    """
    p, d, q = sol.objectives
    results = {}

    def converged(solution):
        # B10: a time-limited solve can find nothing at all.
        return solution is not None and bool((solution.meta or {}).get("optimal"))

    best_profit = solve(inst, "profit", bounds={"duration": d, "projects": q},
                        time_limit=time_limit, backend=backend)
    results["profit"] = (best_profit.profit if best_profit else p, p)

    least_duration = solve(inst, "duration", bounds={"profit": p, "projects": q},
                           time_limit=time_limit, backend=backend)
    results["duration"] = (least_duration.max_duration if least_duration else d, d)

    least_projects = solve(inst, "projects", bounds={"profit": p, "duration": d},
                           time_limit=time_limit, backend=backend)
    results["projects"] = (least_projects.max_projects if least_projects else q, q)

    # A dominating solution is one that was actually FOUND to be strictly
    # better. Any incumbent is feasible, so this conclusion is valid even
    # from a solve that did not converge. The reverse is not: a solve that
    # merely failed to beat the point proves nothing unless it converged.
    ok = (results["profit"][0] <= p
          and results["duration"][0] >= d
          and results["projects"][0] >= q)
    results["proved"] = all(converged(s) for s in
                            (best_profit, least_duration, least_projects))
    return ok, results


def check_front(inst, front, time_limit=None, verbose=True, backend="gurobi"):
    """Run `check_point_nondominated` on every point, and check mutual dominance.

    Returns (failures, summary). A failure means a dominating solution was
    found, which is always a real error. `summary["proved"]` counts the points
    where all three solves converged, so "no dominating solution found" could
    be upgraded to "proved non-dominated".
    """
    failures, proved = [], 0
    for idx, sol in enumerate(front):
        verify(inst, sol)
        ok, detail = check_point_nondominated(inst, sol, time_limit, backend)
        if not ok:
            failures.append((idx, sol.objectives, detail))
        elif detail["proved"]:
            proved += 1
        if verbose:
            mark = "FAIL" if not ok else ("proved" if detail["proved"] else "no counter-example")
            print(f"  point {idx:>2} {str(sol.objectives):>16}  {mark}")

    # no point on the front may dominate another
    for a in front:
        for b in front:
            if a is not b and dominates(a.objectives, b.objectives):
                failures.append(("pairwise", a.objectives, b.objectives))

    summary = {"points": len(front), "proved": proved, "failures": len(failures)}
    if verbose:
        print(f"  -> {proved}/{len(front)} proved non-dominated, "
              f"{len(front) - proved - len(failures)} with no counter-example found, "
              f"{len(failures)} failures")
    return failures, summary


def random_feasible_solutions(inst, n=200, seed=0, time_limit=None, backend="gurobi"):
    """Sample feasible schedules by optimising random objective directions.

    Used to look for a point the front might have missed: if any sampled
    solution is non-dominated with respect to the front, the front is wrong.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from .model import build_model, _run_gurobi, _run_highs

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m, v, f = build_model(inst)
        if time_limit:
            m.Params.TimeLimit = time_limit
        # random linear direction over the raw decision variables, which
        # explores corners a weighted objective sum would never reach
        m.setObjective(gp.quicksum(rng.uniform(-1, 1) * var
                                   for var in v["X"].values())
                       + gp.quicksum(rng.uniform(0, 5) * v["Y"][j] for j in inst["J"]),
                       GRB.MAXIMIZE)
        sol, _ = (_run_gurobi(inst, m, v) if backend == "gurobi"
                  else _run_highs(inst, m, v, time_limit))
        if sol is not None:
            verify(inst, sol)
            out.append(sol)
    return out


def search_for_missed_points(inst, front, n=200, seed=0, time_limit=None, backend="gurobi"):
    """Return any sampled solution that the front fails to dominate."""
    front_points = [s.objectives for s in front]
    missed = []
    for sol in random_feasible_solutions(inst, n, seed, time_limit, backend):
        pt = sol.objectives
        if pt in front_points:
            continue
        if not any(dominates(fp, pt) for fp in front_points):
            missed.append(pt)
    return missed
