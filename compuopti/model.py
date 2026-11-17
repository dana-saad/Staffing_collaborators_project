"""The CompuOpti MILP, with the three objectives used in the multi-objective study.

Decision variables
------------------
X[i,j,k,t]  1 if person i spends day t on qualification k of job j
Y[j]        1 if job j is completed
W[j,t]      1 if anyone works on job j on day t
Z[i,j]      1 if person i works on job j on at least one day
B[j], E[j]  first and last day worked on job j  (0 when j is not completed)
D[j]        duration of job j, E[j] - B[j] + 1  (0 when j is not completed)
L[j]        days of lateness of job j
Tmax        largest D[j]                                -> objective f2
Pmax        largest number of jobs one person works on   -> objective f3

Objectives
----------
f1  profit            sum_j ( g_j Y_j - c_j L_j )           maximise
f2  longest project   Tmax                                   minimise
f3  busiest employee  Pmax                                   minimise

Two formulation choices are worth defending, because they are what make the
model small enough to solve the large instance in seconds:

1.  X[i,j,k,t] exists only when the assignment is possible at all: person i
    holds qualification k, job j needs k, and t is not one of i's leave days.
    That is exactly equivalent to creating every variable and fixing the
    impossible ones to zero - the qualification and leave constraints of the
    written model - but on the large instance it is 4 014 variables instead of
    about 93 000.

2.  The variables W[j,t] and Z[i,j] aggregate what would otherwise be one
    constraint per X variable. Linking start and end dates directly to
    X[i,j,k,t] needs |keys| constraints; routing them through W[j,t] needs
    |J| x |H|. On the medium instance that is the difference between roughly
    4 100 constraints and 1 300.
"""

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

#: objective keys, in the order used everywhere (tuples, tables, plots)
OBJECTIVES = ("profit", "duration", "projects")

#: +1 for objectives to maximise, -1 for objectives to minimise
SENSE = {"profit": +1, "duration": -1, "projects": -1}

LABELS = {
    "profit": "profit",
    "duration": "longest project (days)",
    "projects": "projects per employee (max)",
}

ATTR = {"profit": "profit", "duration": "max_duration", "projects": "max_projects"}


@dataclass
class Solution:
    """A schedule together with its three objective values.

    Objective values are recomputed from `schedule` rather than read off the
    solver's variables, so they are true values of the schedule whatever the
    solver did with the slack left in the auxiliary variables.
    """
    schedule: dict                      # (person, day) -> (job, qualification)
    done: list                          # completed jobs
    start: dict = field(default_factory=dict)
    end: dict = field(default_factory=dict)
    duration: dict = field(default_factory=dict)
    late: dict = field(default_factory=dict)
    per_person: dict = field(default_factory=dict)   # person -> set of jobs
    profit: int = 0
    max_duration: int = 0
    max_projects: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def objectives(self):
        """(profit, longest project, busiest employee) - the point in objective space."""
        return (self.profit, self.max_duration, self.max_projects)

    def __repr__(self):
        return (f"Solution(profit={self.profit}, duration={self.max_duration}, "
                f"projects={self.max_projects}, done={len(self.done)})")


def build_model(inst, verbose=False):
    """Create the MILP and return (model, vars, objective expressions)."""
    H, S, J, h = inst["H"], inst["S"], inst["J"], inst["h"]
    n, g, d, c = inst["n"], inst["g"], inst["d"], inst["c"]

    m = gp.Model("CompuOpti")
    m.Params.OutputFlag = int(verbose)

    keys = [(i, j, k, t)
            for i in S for j in J for k in n[j] if k in inst["QS"][i]
            for t in H if t not in inst["V"][i]]
    pairs = sorted({(i, j) for (i, j, _, _) in keys})
    jt = sorted({(j, t) for (_, j, _, t) in keys})
    capacity = {i: h - len(inst["V"][i]) for i in S}

    X = m.addVars(keys, vtype=GRB.BINARY, name="X")
    Y = m.addVars(J, vtype=GRB.BINARY, name="Y")
    W = m.addVars(jt, vtype=GRB.BINARY, name="W")
    Z = m.addVars(pairs, vtype=GRB.BINARY, name="Z")
    B = m.addVars(J, vtype=GRB.INTEGER, lb=0, ub=h, name="B")
    E = m.addVars(J, vtype=GRB.INTEGER, lb=0, ub=h, name="E")
    D = m.addVars(J, vtype=GRB.INTEGER, lb=0, ub=h, name="D")
    L = m.addVars(J, vtype=GRB.INTEGER, lb=0, ub=h, name="L")
    Tmax = m.addVar(vtype=GRB.INTEGER, lb=0, ub=h, name="Tmax")
    Pmax = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(J), name="Pmax")

    # (C1) a person does at most one job-qualification per day
    m.addConstrs((X.sum(i, "*", "*", t) <= 1 for i in S for t in H), name="one_task")

    # (C2) coverage: a completed job gets exactly the person-days it needs for
    #      each of its qualifications, an abandoned job gets none. Writing this
    #      as an equality merges the statement's two inequalities
    #          Y_j n_jk <= sum X   and   sum X <= n_jk
    #      and additionally stops the solver parking idle days on jobs it never
    #      finishes, which would distort f2 and f3.
    m.addConstrs((X.sum("*", j, k, "*") == n[j][k] * Y[j] for j in J for k in n[j]),
                 name="coverage")

    # (C3) W[j,t] is forced on as soon as anybody works on j that day
    m.addConstrs((gp.quicksum(X[i, j, k, t] for (i, jj, k, tt) in keys
                              if jj == j and tt == t) <= len(S) * W[j, t]
                  for (j, t) in jt), name="worked")

    # (C4) the end date is at least, and the start date at most, every worked day
    m.addConstrs((E[j] >= t * W[j, t] for (j, t) in jt), name="end")
    m.addConstrs((B[j] <= t * W[j, t] + h * (1 - W[j, t]) for (j, t) in jt), name="start")

    # (C5) an abandoned job has no end date and no duration
    m.addConstrs((E[j] <= h * Y[j] for j in J), name="end_zero")
    m.addConstrs((D[j] <= h * Y[j] for j in J), name="dur_zero")

    # (C6) duration, only binding for completed jobs
    m.addConstrs((D[j] >= E[j] - B[j] + 1 - h * (1 - Y[j]) for j in J), name="duration")

    # (C7) lateness
    m.addConstrs((L[j] >= E[j] - d[j] for j in J), name="late")

    # (C8) Z[i,j] is forced on as soon as i spends any day on j
    m.addConstrs((gp.quicksum(X[i, j, k, t] for (ii, jj, k, t) in keys
                              if ii == i and jj == j) <= capacity[i] * Z[i, j]
                  for (i, j) in pairs), name="touch")

    # (C9) Pmax dominates every person's project count
    m.addConstrs((gp.quicksum(Z[i, j] for (ii, j) in pairs if ii == i) <= Pmax
                  for i in S), name="pmax")

    # (C10) Tmax dominates every duration
    m.addConstrs((D[j] <= Tmax for j in J), name="tmax")

    f = {
        "profit": gp.quicksum(g[j] * Y[j] - c[j] * L[j] for j in J),
        "duration": Tmax,
        "projects": Pmax,
    }
    v = {"X": X, "Y": Y, "W": W, "Z": Z, "B": B, "E": E, "D": D, "L": L,
         "Tmax": Tmax, "Pmax": Pmax, "keys": keys, "pairs": pairs, "jt": jt}
    return m, v, f


def extract_solution(inst, v, value=None):
    """Read the schedule out of a solved model and recompute the objectives.

    `value` maps a Gurobi variable to its value; it defaults to reading `.X`,
    and is supplied explicitly when another solver produced the values.
    """
    if value is None:
        def value(var):
            return var.X

    schedule = {(i, t): (j, k) for (i, j, k, t), var in v["X"].items() if value(var) > 0.5}

    by_job = {}
    per_person = {i: set() for i in inst["S"]}
    for (i, t), (j, k) in schedule.items():
        by_job.setdefault(j, []).append(t)
        per_person[i].add(j)

    done = [j for j in inst["J"] if value(v["Y"][j]) > 0.5]
    start = {j: min(by_job[j]) for j in done}
    end = {j: max(by_job[j]) for j in done}
    duration = {j: end[j] - start[j] + 1 for j in done}
    late = {j: max(0, end[j] - inst["d"][j]) for j in done}

    return Solution(
        schedule=schedule,
        done=done,
        start=start, end=end, duration=duration, late=late,
        per_person=per_person,
        profit=sum(inst["g"][j] - inst["c"][j] * late[j] for j in done),
        max_duration=max(duration.values(), default=0),
        max_projects=max((len(p) for p in per_person.values()), default=0),
    )


def verify(inst, sol):
    """Check a solution against the problem statement, independently of the solver.

    Raises AssertionError on any violation. This runs after every solve, so a
    silent modelling error cannot survive into the results.
    """
    worked = {}
    for (i, t), (j, k) in sol.schedule.items():
        # one task per person per day is structural: (i, t) is a dict key
        assert 1 <= t <= inst["h"], f"day {t} outside the horizon"
        assert k in inst["QS"][i], f"{i} does not hold qualification {k}"
        assert t not in inst["V"][i], f"{i} is on leave on day {t}"
        assert k in inst["n"][j], f"{j} does not require qualification {k}"
        assert j in sol.done, f"work scheduled on uncompleted job {j}"
        worked[j, k] = worked.get((j, k), 0) + 1

    for j in sol.done:
        for k, need in inst["n"][j].items():
            got = worked.get((j, k), 0)
            assert got == need, f"{j}/{k}: {got} person-days scheduled, {need} required"

    assert sol.profit == sum(inst["g"][j] - inst["c"][j] * sol.late[j] for j in sol.done)
    assert sol.max_duration == max(sol.duration.values(), default=0)
    assert sol.max_projects == max((len(p) for p in sol.per_person.values()), default=0)
    return True


def _augmented_objective(inst, f, maximise, augment):
    """Main objective plus a tiny term that breaks ties towards the other two.

    Without it, a solver asked only for maximum profit may return a schedule
    that drags on for no reason - any of the equally profitable schedules will
    do. The term makes it return the one that is best on the other objectives,
    which is what stops the epsilon-constraint scan from reporting weakly
    dominated points. This is the augmentation of the AUGMECON method.

    delta is small enough that it can never change the value of the main
    objective: all three objectives are integral, so it is enough that
    delta x (largest possible secondary term) < 1.
    """
    span = (sum(inst["g"].values()) + inst["h"] * sum(inst["c"].values())
            + inst["h"] + len(inst["J"]) + 1)
    delta = 0.5 / span if augment else 0.0
    secondary = gp.quicksum(-SENSE[k] * f[k] for k in OBJECTIVES if k != maximise)
    return f[maximise] - SENSE[maximise] * delta * secondary


def solve(inst, maximise="profit", bounds=None, time_limit=None, verbose=False,
          augment=True, mip_gap=0.0, backend="gurobi"):
    """Solve one scalarised problem.

    maximise    which objective to optimise ("profit", "duration", "projects");
                objectives whose SENSE is -1 are minimised
    bounds      optional epsilon-constraints, e.g. {"duration": 6, "projects": 3}
                meaning longest project <= 6 and projects per employee <= 3
    backend     "gurobi", or "highs" to solve the same model with the open
                source HiGHS solver (used to cross-check results, and to run
                instances larger than a size-limited Gurobi licence allows)

    Returns a Solution, or None if no solution was found.
    """
    m, v, f = build_model(inst, verbose)
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap

    for key, eps in (bounds or {}).items():
        if SENSE[key] < 0:
            m.addConstr(f[key] <= eps, name=f"eps_{key}")
        else:
            m.addConstr(f[key] >= eps, name=f"eps_{key}")

    obj = _augmented_objective(inst, f, maximise, augment)
    m.setObjective(obj, GRB.MAXIMIZE if SENSE[maximise] > 0 else GRB.MINIMIZE)

    if backend == "gurobi":
        sol, meta = _run_gurobi(inst, m, v)
    elif backend == "highs":
        sol, meta = _run_highs(inst, m, v, time_limit, mip_gap)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if sol is None:
        return None
    verify(inst, sol)
    meta.update(maximise=maximise, bounds=dict(bounds or {}), backend=backend)
    sol.meta = meta
    return sol


def _run_gurobi(inst, m, v):
    m.optimize()
    if m.SolCount == 0:
        return None, {}
    sol = extract_solution(inst, v)
    return sol, {"status": m.Status, "gap": m.MIPGap, "runtime": m.Runtime,
                 "n_vars": m.NumVars, "n_constrs": m.NumConstrs}


def _run_highs(inst, m, v, time_limit=None, mip_gap=0.0):
    """Solve the Gurobi-built model with HiGHS, via a temporary MPS file.

    Building and writing a model is always allowed; only optimising is subject
    to licence limits. Variables are renamed to positional names first, because
    the indexed names Gurobi generates do not survive an MPS round trip.
    """
    import os
    import tempfile
    import highspy

    m.update()
    gvars = m.getVars()
    original = [var.VarName for var in gvars]
    for idx, var in enumerate(gvars):
        var.VarName = f"c{idx}"
    for idx, con in enumerate(m.getConstrs()):
        con.ConstrName = f"r{idx}"
    m.update()

    path = tempfile.mktemp(suffix=".mps")
    m.write(path)
    for var, nm in zip(gvars, original):      # restore, the caller may reuse m
        var.VarName = nm
    m.update()

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.readModel(path)
    h.run()
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    values = list(h.getSolution().col_value)
    os.unlink(path)

    if not values or status in ("Infeasible", "Empty"):
        return None, {}

    lookup = {var.VarName: values[idx] for idx, var in enumerate(gvars)}
    sol = extract_solution(inst, v, value=lambda var: lookup[var.VarName])
    return sol, {"status": status, "gap": info.mip_gap, "runtime": h.getRunTime(),
                 "n_vars": m.NumVars, "n_constrs": m.NumConstrs}


def model_size(inst):
    """Variable and constraint counts, without solving."""
    m, _, _ = build_model(inst)
    m.update()
    return {"n_vars": m.NumVars, "n_constrs": m.NumConstrs}
