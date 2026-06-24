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
