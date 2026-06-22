import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def plot_convergence(results, title="Solver Convergence"):
    # same algo gets same color, multiple runs stack naturally
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = list(dict.fromkeys(r.algo for r in results))  # deduplicate, preserve order
    colors = plt.cm.tab10.colors
    color_of = {algo: colors[i % len(colors)] for i, algo in enumerate(algos)}

    seen = set()
    for result in results:
        label = result.algo if result.algo not in seen else "_nolegend_" # one legend entry per algo
        seen.add(result.algo)
        n  = len(result.convergence)
        xs = [i / max(n - 1, 1) for i in range(n)] # normalize to [0,1] so SA(50k) and ACO(500) are comparable
        ax.plot(xs, result.convergence, color=color_of[result.algo],
                label=label, linewidth=1.5, alpha=0.75)

    ax.set_xlabel("Fraction of iterations")
    ax.set_ylabel("Best score so far")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_convergence_comparison(results, title="Convergence Comparison (mean ± std)"):
    # multiple runs of multiple algos to compare side by side
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r.algo].append(r.convergence)

    algos = list(by_algo.keys())
    colors = plt.cm.tab10.colors
    color_of = {algo: colors[i % len(colors)] for i, algo in enumerate(algos)}

    fig, ax = plt.subplots(figsize=(10, 5))

    for algo, runs in by_algo.items():
        max_len = max(len(r) for r in runs)
        padded = [r + [r[-1]] * (max_len - len(r)) for r in runs]  
        arr = np.array(padded) # shape: (n_runs, n_iterations)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = color_of[algo]

        xs = [i / max(max_len - 1, 1) for i in range(max_len)]  # normalize so all algos share the same x scale
        ax.plot(xs, mean, color=color, label=algo, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Fraction of iterations")
    ax.set_ylabel("Best score so far")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
