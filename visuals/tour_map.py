import matplotlib.pyplot as plt

def plot_tour(instance, result, title="Best Tour"):
    tour = result.solution.tour
    packing = result.solution.packing
    cities = instance.cities

    # close the loop
    xs = [cities[c][0] for c in tour] + [cities[tour[0]][0]]
    ys = [cities[c][1] for c in tour] + [cities[tour[0]][1]]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(xs, ys, color="steelblue", linewidth=1, zorder=1)
    ax.scatter([c[0] for c in cities], [c[1] for c in cities], color="steelblue", s=30, zorder=2)

    # cities where at least one item was picked 
    picked_cities = {instance.items[i].city for i in range(instance.m) if packing[i] == 1}
    ax.scatter([cities[c][0] for c in picked_cities],
               [cities[c][1] for c in picked_cities],
               color="tomato", s=80, zorder=3, label="item picked")

    ax.scatter([cities[tour[0]][0]], [cities[tour[0]][1]], color="gold", s=120, zorder=4, label="start")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    return fig
