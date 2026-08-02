def normalize_tour(tour):
    idx = tour.index(0)
    return tour[idx:] + tour[:idx]
