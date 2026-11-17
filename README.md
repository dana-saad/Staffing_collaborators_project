# CompuOpti — staffing collaborators on projects

Decision Modelling project, MSc AI, CentraleSupélec 2026–27.

CompuOpti has a small staff of engineers, each holding some qualifications and
with leave days booked, and a list of client projects. Each project needs a
given number of person-days of each of several qualifications, pays a gain only
if it is **completely** finished, and costs a penalty for every day it runs past
its due date. Not everything can be staffed, so some projects have to be
dropped.

This repository formulates that as a multi-objective MILP, computes the complete
set of non-dominated solutions, and visualises them in both objective and
decision space.

## Start here

`notebooks/compuopti.ipynb` is the whole study, in order, with the reasoning
behind every modelling choice written out. Everything below is the machinery it
calls.

## The model in one page

**Decision variables** — `X[i,j,k,t]` is the real decision: person *i* spends
day *t* on qualification *k* of job *j*. `Y[j]` marks a completed job; `W[j,t]`,
`Z[i,j]`, `B[j]`, `E[j]`, `D[j]`, `L[j]`, `Tmax`, `Pmax` are bookkeeping that
lets the objectives stay linear.

**Three objectives**

| | | |
|---|---|---|
| `f1` | profit, `sum_j (g_j Y_j − c_j L_j)` | maximise |
| `f2` | the longest single project, `Tmax` | minimise |
| `f3` | the busiest employee's project count, `Pmax` | minimise |

They conflict: concentrating people finishes a project fast but needs many of
them on it; spreading them out does the reverse; insisting on either usually
costs profit.

**Method** — the ε-constraint method. Profit is maximised subject to
`f2 ≤ ε2` and `f3 ≤ ε3`, sweeping every integer pair. Because `f2` and `f3` are
integers with small ranges, the sweep is complete, and so is the resulting
front. A weighted sum would be cheaper but can only ever return solutions on the
convex hull of the front; §5.4 of the notebook measures what that loses.

Most grid cells never need a solve: if a looser cell has been solved and its
solution respects the tighter bounds, it is optimal there too. In practice that
removes 80–90% of the grid.

## Layout

```
compuopti/
    instance.py   loading, JSON round-trip, and the random instance generator
    model.py      the MILP, both solver backends, and the feasibility checker
    pareto.py     dominance, the payoff table, epsilon-constraint, weighted sum
    check.py      independent verification of a computed front
    viz.py        schedule plots (decision space), front plots (objective space)
instances/        the three given instances; generated/ holds produced ones
notebooks/        compuopti.ipynb — the study
scripts/          run_fronts.py, performance.py, build_notebook.py
results/          cached fronts and performance measurements
```

## Running it

```bash
pip install gurobipy matplotlib numpy
jupyter notebook notebooks/compuopti.ipynb
```

Gurobi's pip package ships a size-limited licence capped at 2 000 variables.
The toy and medium instances fit; the large instance (5 054 variables) needs a
free academic licence from gurobi.com.

Where no Gurobi licence is available, every entry point takes `backend="highs"`
and solves the same model with the open-source HiGHS solver instead:

```python
co.solve(inst, "profit", backend="highs")
co.epsilon_constraint(inst, backend="highs")
```

HiGHS is several times slower here but has no size limit, so it is also how the
results in this repository were cross-checked against a second solver.

To recompute from scratch:

```bash
python scripts/run_fronts.py instances/medium_instance.json results/medium_front.pkl
python scripts/performance.py
python scripts/build_notebook.py          # regenerates the notebook from source
```

## Results

| Instance | Non-dominated solutions | Best profit | Solves / grid cells | Exact |
|---|---|---|---|---|
| toy | 10 | 65 | 11 / 36 | yes |
| medium | 43 | 413 | 47 / 368 | yes |
| large | 10 found | 790 (true optimum 817) | 18 / 266 | no — see below |

The large model has 5 054 variables, past the cap of the licence that ships with
`pip install gurobipy`, so that run used HiGHS with the grid coarsened to every
second value and a 60 s cap per solve. Every point it returns is genuinely
non-dominated, but points between grid lines are missed and a capped solve can
stop short of the optimum. On a full academic licence, run it exactly:

```python
co.epsilon_constraint(large, step=1, backend="gurobi")
```

## Correctness

Three checks, all in the notebook:

1. **Every schedule is feasible.** `model.verify` re-reads each solution and
   re-checks it against the problem statement — qualifications held, leave
   respected, coverage exact — and recomputes all three objective values from
   the schedule rather than trusting the solver's variables. It runs after
   every solve.
2. **Every reported point is non-dominated.** `check.check_front` tests it with
   three further solves per point. All 10 toy points and a 6-point sample of
   the medium front came back **proved**, with no failures.
3. **Nothing was missed.** Completeness is argued in the notebook and tested by
   generating feasible schedules in random directions and confirming the front
   dominates all of them — 300 samples on toy, none missed.

One trap is worth knowing about, because the first version of check 2 fell into
it. Asking whether the *optimum* of each solve equals the point is wrong under a
time limit: a solve can stop early and report a value worse than the point's
own, which looks like a contradiction and is really an unfinished search. On the
medium instance it flagged `(250, 3, 5)` as failing, reporting a best duration of
4 after timing out with a 52 % gap — while forcing `duration ≤ 3` solves to
proven optimality and returns the point itself. The test is now "no solve
*found* anything strictly better", which is valid even from a non-converged
solve because any incumbent is feasible, and `check_front` reports "proved"
separately from "no counter-example found".
