import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from core.ttp_evaluator import TTPEvaluator


def _normalize_tour(tour):
    idx = tour.index(0)
    return tour[idx:] + tour[:idx]


def plot_tour(instance, result, ax=None, title="Best Tour"):
    tour = _normalize_tour(result.solution.tour)
    packing = result.solution.packing
    cities = instance.cities

    # per-city totals for picked items
    city_weight = defaultdict(float)
    city_profit = defaultdict(float)
    for item in instance.items:
        if packing[item.id] == 1:
            city_weight[item.city] += item.weight
            city_profit[item.city] += item.profit

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # tour path (closed loop)
    xs = [cities[c][0] for c in tour] + [cities[tour[0]][0]]
    ys = [cities[c][1] for c in tour] + [cities[tour[0]][1]]
    ax.plot(xs, ys, color="steelblue", linewidth=1, zorder=1)

    # base dots for all cities
    ax.scatter([c[0] for c in cities], [c[1] for c in cities],
               color="steelblue", s=30, zorder=2)

    # cities with picked items: size scaled by weight, labelled with w= p=
    if city_weight:
        max_w = max(city_weight.values())
        for city, w in city_weight.items():
            size = 60 + 300 * (w / max_w)
            ax.scatter(cities[city][0], cities[city][1],
                       color="tomato", s=size, zorder=3)
            ax.annotate(
                f"w={w:.0f}\np={city_profit[city]:.0f}",
                xy=(cities[city][0], cities[city][1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="darkred",
            )

    # start city
    ax.scatter([cities[tour[0]][0]], [cities[tour[0]][1]],
               color="gold", s=120, zorder=4)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                   markersize=8, label="item picked (size ∝ weight)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                   markersize=8, label="start"),
    ])

    if own_fig:
        plt.tight_layout()
        return fig


def plot_tour_with_weight_profit(instance, result, algo_name=""):
    evaluator = TTPEvaluator(instance)
    _, trace = evaluator.evaluate_traced(result.solution)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
    plot_tour(instance, result, ax=ax1, title=f"Best tour {algo_name}".strip())

    steps = [row for row in trace if row["distance"] is not None]
    cum_dist, d = [], 0.0
    for row in steps:
        d += row["distance"]
        cum_dist.append(d)

    ax2b = ax2.twinx()
    ax2.plot(cum_dist, [row["weight"] for row in steps], color="darkorange", linewidth=2)
    ax2b.plot(cum_dist, [row["profit_so_far"] for row in steps], color="seagreen", linewidth=2)
    ax2.set_xlabel("Cumulative distance")
    ax2.set_ylabel("Weight carried", color="darkorange")
    ax2b.set_ylabel("Profit so far", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2b.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_title("Weight & Profit by distance (best run)")
    ax2.legend(handles=[
        plt.Line2D([0], [0], color="darkorange", lw=2, label="weight"),
        plt.Line2D([0], [0], color="seagreen", lw=2, label="profit"),
    ])
    plt.tight_layout()
    return fig, trace


def trace_to_dataframe(trace):
    return pd.DataFrame([
        {
            "step": i,
            "city": row["city"],
            "next_city": row["next_city"],
            "items_picked": row["items_picked"] or [],
            "weight": row["weight"],
            "profit_so_far": row["profit_so_far"],
            "distance": row["distance"],
            "time": row["time"],
            "objective_so_far": row["objective_so_far"],
        }
        for i, row in enumerate(trace)
    ]).set_index("step")


def results_summary(results):
    scores = [r.convergence[-1] for r in results]
    runtimes = [r.runtime for r in results]
    return pd.DataFrame({
        "score": scores,
        "runtime_s": runtimes,
    }, index=pd.RangeIndex(1, len(results) + 1, name="run"))
