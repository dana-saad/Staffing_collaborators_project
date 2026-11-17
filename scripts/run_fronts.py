"""Compute and store the Pareto front of one instance.

    python scripts/run_fronts.py <instance.json> <out.pkl> [backend] [time_limit]
"""
import pickle
import sys
import time

sys.path.insert(0, ".")
import compuopti as co


def main():
    path, out = sys.argv[1], sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else "gurobi"
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else None

    inst = co.load_instance(path)
    print(co.instance_stats(inst), flush=True)
    print(co.model_size(inst), flush=True)

    t0 = time.time()
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    r = co.epsilon_constraint(inst, backend=backend, time_limit=tl, progress=True, step=step)
    r["instance_path"] = path
    r["backend"] = backend
    r["time_limit"] = tl
    with open(out, "wb") as f:
        pickle.dump(r, f)

    print(co.front_table(r["front"], inst), flush=True)
    print(f"solves={r['n_solves']} reused={r['n_reused']} cells={r['n_cells']} "
          f"wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
