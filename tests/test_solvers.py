from core.ttp_loader import load_instance
from solvers.aco.aco_solver import ACOSolver

INSTANCES = [
    "benchmarks/eil51/eil51_n50_uncorr_01.ttp",
    "benchmarks/eil51/eil51_n50_bounded-strongly-corr_01.ttp",
]

SOLVERS = [
    ("ACO", lambda: ACOSolver(max_iterations=500)),
]

for path in INSTANCES:
    instance = load_instance(path)
    name = path.split("/")[-1]
    print(f"\n{name}")

    for algo_name, make_solver in SOLVERS:
        result = make_solver().solve(instance)
        print(f"{algo_name:6} score={result.convergence[-1]:.2f} runtime={result.runtime:.2f}s")