import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(results, title="Solver Convergence"):
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = list(dict.fromkeys(r.algo for r in results))
    colors = plt.cm.tab10.colors
    color_of = {algo: colors[i % len(colors)] for i, algo in enumerate(algos)}

    seen = set()
    for result in results:
        label = result.algo if result.algo not in seen else "_nolegend_"
        seen.add(result.algo)
        n = len(result.convergence)
        xs = [i / max(n - 1, 1) for i in range(n)]
        ax.plot(xs, result.convergence, color=color_of[result.algo],
                label=label, linewidth=1.5, alpha=0.75)

    ax.set_xlabel("Fraction of iterations")
    ax.set_ylabel("Best score so far")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_convergence_comparison(all_results, title="Solver Convergence", n_points=200):
    grid = np.linspace(0.0, 1.0, n_points)
    colors = plt.cm.tab10.colors

    stats = {}
    for algo, results in all_results.items():
        curves = []
        for r in results:
            n = len(r.convergence)
            xs = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0, 1.0])
            ys = r.convergence if n > 1 else [r.convergence[0], r.convergence[0]]
            curves.append(np.interp(grid, xs, ys))
        curves = np.array(curves)
        stats[algo] = (curves.mean(axis=0), curves.std(axis=0))

    ordered = sorted(stats.items(), key=lambda kv: kv[1][0][-1], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (algo, (mean, std)) in enumerate(ordered):
        color = colors[i % len(colors)]
        ax.plot(grid, mean, color=color, linewidth=2, label=algo)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Fraction of iterations")
    ax.set_ylabel("Best score so far")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
