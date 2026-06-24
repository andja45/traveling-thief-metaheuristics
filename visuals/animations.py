import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker

def _normalize_tour(tour):
    idx = tour.index(0)
    return tour[idx:] + tour[:idx]


def animate_tour(instance, result, interval=100):
    # draws the tour edge by edge
    tour = _normalize_tour(result.solution.tour)
    cities = instance.cities
    n = len(tour)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.scatter([c[0] for c in cities], [c[1] for c in cities], color="steelblue", s=30, zorder=2)

    line, = ax.plot([], [], color="steelblue", linewidth=1.5)

    def update(frame):
        pts = tour[:frame + 1]
        line.set_data([cities[c][0] for c in pts], [cities[c][1] for c in pts])
        return line,

    return animation.FuncAnimation(
        fig, update, frames=n + 1, interval=interval, blit=True
    )


def plot_pheromone(tau, title="Pheromone Matrix"):
    # static heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.array(tau), cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, label="τ (pheromone level)")
    ax.set_title(title)
    ax.set_xlabel("City j")
    ax.set_ylabel("City i")
    plt.tight_layout()
    return fig


def animate_pheromone(tau_history, interval=200):
    # animated heatmap over iterations
    arr  = [np.array(t) for t in tau_history]
    vmin = min(a.min() for a in arr)
    vmax = max(a.max() for a in arr)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr[0], cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="τ")
    label = ax.set_title("Iteration 0")
    ax.set_xlabel("City j")
    ax.set_ylabel("City i")

    def update(frame):
        im.set_array(arr[frame])
        label.set_text(f"Iteration {frame}")
        return im, label

    return animation.FuncAnimation(
        fig, update, frames=len(arr), interval=interval, blit=False
    )


def animate_weight_profit(instance, trace, interval=120):
    # build cumulative distance + per-step weight/profit arrays
    # last trace row has distance=None (final city, no travel), skip for x-axis
    steps = [row for row in trace if row["distance"] is not None]

    cum_dist = []
    d = 0.0
    for row in steps:
        d += row["distance"]
        cum_dist.append(d)

    weights = [row["weight"] for row in steps]
    profits = [row["profit_so_far"] for row in steps]
    n = len(cum_dist)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.set_xlabel("Cumulative distance")
    ax1.set_ylabel("Weight carried", color="darkorange")
    ax2.set_ylabel("Profit so far", color="seagreen")
    ax1.tick_params(axis="y", labelcolor="darkorange")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax1.set_xlim(0, cum_dist[-1] * 1.02)
    ax1.set_ylim(0, max(weights) * 1.15 if weights else 1)
    ax2.set_ylim(0, max(profits) * 1.15 if profits else 1)

    line_w, = ax1.plot([], [], color="darkorange", linewidth=2, label="weight")
    line_p, = ax2.plot([], [], color="seagreen",   linewidth=2, label="profit")
    vline   = ax1.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    title   = ax1.set_title("Step 0")

    lines = [line_w, line_p]
    fig.legend(lines, ["weight", "profit"], loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()

    def update(frame):
        f = frame + 1
        line_w.set_data(cum_dist[:f], weights[:f])
        line_p.set_data(cum_dist[:f], profits[:f])
        vline.set_xdata([cum_dist[frame]])
        city = steps[frame]["city"]
        title.set_text(f"Step {f}/{n}  city={city}  "
                       f"w={weights[frame]:.1f}  p={profits[frame]:.1f}")
        return line_w, line_p, vline, title

    return animation.FuncAnimation(
        fig, update, frames=n, interval=interval, blit=False
    )


def animate_wolves(scores_history, interval=150):
    # shows wolf score distribution collapsing toward alpha each iteration
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Wolf rank")
    ax.set_ylabel("Score")

    n_wolves = len(scores_history[0])
    scat = ax.scatter(range(n_wolves), scores_history[0], c="steelblue", s=20)
    title = ax.set_title("Iteration 0")

    all_scores = [s for frame in scores_history for s in frame]
    ax.set_ylim(min(all_scores) * 0.99, max(all_scores) * 1.01)

    def update(frame):
        scat.set_offsets(list(enumerate(scores_history[frame])))
        title.set_text(f"Iteration {frame}")
        return scat, title

    return animation.FuncAnimation(
        fig, update, frames=len(scores_history), interval=interval, blit=False
    )
