import csv
import os
from core.ttp_loader import load_instance
from solvers.aco_solver import ACOSolver
from solvers.gwo_solver import GWOSolver
from solvers.ga_solver import GA
from solvers.ga_improved_solver import GAImproved
from solvers.sa_solver import SA
from solvers.sa_improved_solver import SAImproved
from solvers.s5_solver import S5Solver
from visuals.convergence import plot_convergence

INSTANCES = {
    "tiny_n5": "benchmarks/tiny/tiny_n5_m4.ttp",
    "tiny_n7": "benchmarks/tiny/tiny_n7_m6.ttp",
    "tiny_n9": "benchmarks/tiny/tiny_n9_m8.ttp",
    "eil51_uncorr": "benchmarks/eil51/eil51_n50_uncorr_01.ttp",
    "eil51_corr": "benchmarks/eil51/eil51_n50_bounded-strongly-corr_01.ttp",
    "pr76_n75_uncorr": "benchmarks/pr76-ttp/pr76_n75_uncorr_01.ttp",
    "pr76_n75_corr": "benchmarks/pr76-ttp/pr76_n75_bounded-strongly-corr_01.ttp",
    "rat195_uncorr": "benchmarks/rat195/rat195_n194_uncorr_01.ttp",
    "rat195_corr": "benchmarks/rat195/rat195_n194_bounded-strongly-corr_01.ttp",
}

SOLVERS = {
    "ACO": lambda: ACOSolver(max_iterations=500),
    "GWO": lambda: GWOSolver(max_iterations=500),
    "GA": lambda: GA(max_iterations=500),
    "GAImproved": lambda: GAImproved(max_iterations=500),
    "SA": lambda: SA(),
    "SAImproved": lambda: SAImproved(),
    "S5": lambda: S5Solver(),
}

RUNS = 3

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
print("\nplot saved to results/")
