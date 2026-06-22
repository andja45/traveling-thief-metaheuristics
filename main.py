import csv
import os
from core.ttp_loader import load_instance
from solvers.aco.aco_solver import ACOSolver
from solvers.gwo.gwo_solver import GWOSolver
from solvers.ga import GA
from solvers.sa import SA
from visuals.convergence import plot_convergence, plot_convergence_comparison

INSTANCES = {
    "eil51_uncorr": "benchmarks/eil51/eil51_n50_uncorr_01.ttp",
    "eil51_corr": "benchmarks/eil51/eil51_n50_bounded-strongly-corr_01.ttp",
}

SOLVERS = {
    "ACO": lambda: ACOSolver(max_iterations=500),
    "GWO": lambda: GWOSolver(max_iterations=500),
    "GA": lambda: GA(max_iterations=500),
    "SA": lambda: SA(),  # uses its own default
}

RUNS = 5

os.makedirs("results", exist_ok=True)
all_results = []

with open("results/all.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["algo", "instance", "run", "score", "runtime"])

    for inst_name, inst_path in INSTANCES.items():
        instance = load_instance(inst_path)
        print(f"\n=== {inst_name} ===")

        for algo_name, make_solver in SOLVERS.items():
            for run in range(1, RUNS + 1):
                result = make_solver().solve(instance)
                all_results.append(result)
                score = result.convergence[-1]
                writer.writerow([algo_name, inst_name, run, f"{score:.2f}", f"{result.runtime:.2f}"])
                print(f"  {algo_name} run {run}: score={score:.2f} time={result.runtime:.2f}s")

plot_convergence(all_results).savefig("results/convergence_raw.png", dpi=150)
plot_convergence_comparison(all_results).savefig("results/convergence_comparison.png", dpi=150)
print("\nplots saved to results/")
