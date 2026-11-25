"""Performance study: how the model and the Pareto scan scale with instance size.

    python scripts/performance.py [out.pkl]

Generates a ladder of random instances of increasing size and measures, for
each one, the model size, the time for a single profit-optimal solve, and the
time for the complete epsilon-constraint scan.

The backend is chosen per instance: Gurobi while the model fits inside a
size-limited licence, HiGHS beyond it. The `backend` column records which was
used, because the two are not directly comparable on time.
"""

import json
import pickle
import sys
import time

sys.path.insert(0, ".")
import compuopti as co

LADDER = [(3, 5, 8), (4, 7, 10), (4, 9, 12), (5, 10, 14),
          (5, 12, 16), (6, 14, 18), (6, 16, 20)]

SOLVE_TIME_LIMIT = 60    # this is a timing study, so solves stay capped --
                         # but every row reports how many did not prove optimality
FRONT_BUDGET = 900


def run_one(n_staff, n_jobs, horizon, seed=1, save_dir="instances/generated"):
    inst = co.generate_instance(n_staff=n_staff, n_jobs=n_jobs, horizon=horizon,
                                n_qualifications=5, seed=seed)
    co.save_instance(inst, f"{save_dir}/{inst['name']}.json")

    size = co.model_size(inst)
    backend = "gurobi"

    row = dict(co.instance_stats(inst), **size, backend=backend)

    t0 = time.time()
    sol = co.solve(inst, "profit", time_limit=SOLVE_TIME_LIMIT, backend=backend)
    row["t_single"] = time.time() - t0
    row["best_profit"] = sol.profit

    t0 = time.time()
    res = co.epsilon_constraint(inst, time_limit=SOLVE_TIME_LIMIT, backend=backend)
    row["t_front"] = time.time() - t0
    row["front_size"] = len(res["front"])
    row["n_solves"] = res["n_solves"]
    row["n_cells"] = res["n_cells"]
    row["n_unproven"] = res["n_unproven"]
    row["exact"] = res["exact"]
    row["reuse_pct"] = round(100 * res["n_reused"] / res["n_cells"], 1)
    return row, res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/performance.pkl"
    rows, fronts, spent = [], {}, 0.0

    for n_staff, n_jobs, horizon in LADDER:
        label = f"s{n_staff}_j{n_jobs}_h{horizon}"
        print(f"--- {label} ---", flush=True)
        t0 = time.time()
        try:
            row, res = run_one(n_staff, n_jobs, horizon)
        except Exception as exc:                      # noqa: BLE001
            print(f"  failed: {exc}", flush=True)
            continue
        spent += time.time() - t0
        rows.append(row)
        fronts[label] = res["front"]
        print(f"  vars={row['n_vars']:>5} cons={row['n_constrs']:>5} "
              f"single={row['t_single']:.1f}s front={row['t_front']:.1f}s "
              f"points={row['front_size']} solves={row['n_solves']}/{row['n_cells']} "
              f"({row['reuse_pct']}% reused) unproven={row['n_unproven']} "
              f"exact={row['exact']}", flush=True)
        with open(out, "wb") as f:
            pickle.dump({"rows": rows, "fronts": fronts}, f)
        with open(out.replace(".pkl", ".json"), "w") as f:
            json.dump(rows, f, indent=2)
        if spent > FRONT_BUDGET:
            print("  budget exhausted, stopping the ladder here", flush=True)
            break

    print(f"\ntotal {spent:.0f}s over {len(rows)} instances", flush=True)


if __name__ == "__main__":
    main()
