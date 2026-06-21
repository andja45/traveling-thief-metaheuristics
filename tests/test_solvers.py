import csv
import os

from core.ttp_loader import load_instance
from solvers.aco.aco_solver import ACOSolver

INSTANCES = [
    "benchmarks/eil51/eil51_n50_uncorr_01.ttp",
    "benchmarks/eil51/eil51_n50_bounded-strongly-corr_01.ttp",
]

SOLVERS = [
    ("ACO", lambda: ACOSolver(max_iterations=500)),
]

RUNS = 5
RESULTS_FILE = "results/aco_packing_swap.csv"

os.makedirs("results", exist_ok=True)

with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if os.path.getsize(RESULTS_FILE) == 0:
        writer.writerow(["algo", "instance", "score", "runtime"])

    for path in INSTANCES:
        instance = load_instance(path)
        name = path.split("/")[-1]
        print(f"\n--- {name} ---")

        for algo_name, make_solver in SOLVERS:
            for run in range(RUNS):
                result = make_solver().solve(instance)
                score = result.convergence[-1]
                writer.writerow([algo_name, name, f"{score:.2f}", f"{result.runtime:.2f}"])
                print(f"  {algo_name} run {run + 1}: score={score:.2f}  runtime={result.runtime:.2f}s")
