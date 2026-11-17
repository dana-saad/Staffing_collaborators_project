"""Instance loading and random instance generation for the CompuOpti problem.

An instance is held as a plain dict whose keys are the symbols used in the
mathematical model, so that the model code reads like the formulation:

    h    time horizon (int)                     H    {1, ..., h}
    Q    set of qualifications                  S    list of staff names
    QS   skills of each person      Q_i^S       V    leave days of each person   V_i
    J    list of job names
    n    required person-days       n_{j,k}     g    gain of a job               g_j
    d    due date                   d_j         c    daily lateness penalty      c_j
"""

import json
import random


def load_instance(path):
    """Read a JSON instance file into the parameter dict described above."""
    with open(path) as f:
        raw = json.load(f)
    return _from_raw(raw)


def _from_raw(raw):
    return {
        "name": raw.get("name", "instance"),
        "h": raw["horizon"],
        "H": list(range(1, raw["horizon"] + 1)),
        "Q": list(raw["qualifications"]),
        "S": [s["name"] for s in raw["staff"]],
        "QS": {s["name"]: set(s["qualifications"]) for s in raw["staff"]},
        "V": {s["name"]: set(s["vacations"]) for s in raw["staff"]},
        "J": [j["name"] for j in raw["jobs"]],
        "n": {j["name"]: dict(j["working_days_per_qualification"]) for j in raw["jobs"]},
        "g": {j["name"]: j["gain"] for j in raw["jobs"]},
        "d": {j["name"]: j["due_date"] for j in raw["jobs"]},
        "c": {j["name"]: j["daily_penalty"] for j in raw["jobs"]},
    }


def to_raw(inst):
    """Inverse of _from_raw: produce the JSON-serialisable form."""
    return {
        "name": inst.get("name", "instance"),
        "horizon": inst["h"],
        "qualifications": inst["Q"],
        "staff": [{"name": i,
                   "qualifications": sorted(inst["QS"][i]),
                   "vacations": sorted(inst["V"][i])} for i in inst["S"]],
        "jobs": [{"name": j,
                  "gain": inst["g"][j],
                  "due_date": inst["d"][j],
                  "daily_penalty": inst["c"][j],
                  "working_days_per_qualification": inst["n"][j]} for j in inst["J"]],
    }


def save_instance(inst, path):
    with open(path, "w") as f:
        json.dump(to_raw(inst), f, indent=4)


# ----------------------------------------------------------------------------
# Instance generator
# ----------------------------------------------------------------------------

#: first names used for generated staff, recycled with a suffix when exhausted
_NAMES = ["Olivia", "Liam", "Emma", "Noah", "Amelia", "Oliver", "Ava", "Elijah",
          "Sophia", "Lucas", "Isabella", "Mason", "Mia", "Ethan", "Charlotte",
          "James", "Harper", "Benjamin", "Evelyn", "Henry"]


def generate_instance(n_staff=5, n_jobs=10, n_qualifications=5, horizon=20,
                      skills_per_person=(2, 4), quals_per_job=(1, 3),
                      days_per_qual=(1, 5), gain=(10, 80), penalty=3,
                      vacation_rate=0.10, due_date_window=(0.5, 1.0),
                      seed=None, name=None):
    """Build a random instance in the same format as the course files.

    The parameters that matter for difficulty are `n_staff`, `n_jobs` and
    `horizon`; the rest shape how tight the instance is.

    skills_per_person  how many qualifications each person holds (min, max)
    quals_per_job      how many qualifications a job needs (min, max)
    days_per_qual      person-days a job needs per qualification (min, max)
    gain               job gain drawn uniformly in this range (min, max)
    vacation_rate      fraction of the horizon each person is on leave
    due_date_window    due dates drawn in this fraction-of-horizon window

    Every qualification is guaranteed to be held by at least one person, so a
    job is never impossible for lack of a skill. Instances can still be partly
    unachievable through lack of capacity, which is the interesting case.
    """
    rng = random.Random(seed)

    quals = [chr(ord("A") + k) for k in range(n_qualifications)]
    names = [_NAMES[i] if i < len(_NAMES) else f"{_NAMES[i % len(_NAMES)]}{i // len(_NAMES) + 1}"
             for i in range(n_staff)]

    # --- staff ---------------------------------------------------------
    staff_skills = {}
    for i in names:
        k = rng.randint(*skills_per_person)
        staff_skills[i] = set(rng.sample(quals, min(k, n_qualifications)))
    # make sure no qualification is orphaned
    for q in quals:
        if not any(q in staff_skills[i] for i in names):
            staff_skills[rng.choice(names)].add(q)

    n_vac = int(round(vacation_rate * horizon))
    vacations = {i: set(rng.sample(range(1, horizon + 1), n_vac)) if n_vac else set()
                 for i in names}

    # --- jobs ----------------------------------------------------------
    jobs = [f"Job{j + 1}" for j in range(n_jobs)]
    need, gains, dues, pens = {}, {}, {}, {}
    lo, hi = due_date_window
    for j in jobs:
        k = rng.randint(*quals_per_job)
        chosen = rng.sample(quals, min(k, n_qualifications))
        need[j] = {q: rng.randint(*days_per_qual) for q in chosen}
        gains[j] = rng.randrange(gain[0], gain[1] + 1, 5)
        dues[j] = rng.randint(max(1, int(lo * horizon)), max(1, int(hi * horizon)))
        pens[j] = penalty

    return {
        "name": name or f"gen_s{n_staff}_j{n_jobs}_h{horizon}_seed{seed}",
        "h": horizon,
        "H": list(range(1, horizon + 1)),
        "Q": quals,
        "S": names,
        "QS": staff_skills,
        "V": vacations,
        "J": jobs,
        "n": need,
        "g": gains,
        "d": dues,
        "c": pens,
    }


def instance_stats(inst):
    """Summary figures used in the performance tables."""
    total_days = sum(sum(v.values()) for v in inst["n"].values())
    capacity = sum(inst["h"] - len(inst["V"][i]) for i in inst["S"])
    return {
        "instance": inst.get("name", "?"),
        "staff": len(inst["S"]),
        "jobs": len(inst["J"]),
        "quals": len(inst["Q"]),
        "horizon": inst["h"],
        "demand": total_days,
        "capacity": capacity,
        "load": round(total_days / capacity, 2) if capacity else float("inf"),
        "total_gain": sum(inst["g"].values()),
    }
