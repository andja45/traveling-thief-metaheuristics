import time
from abc import ABC, abstractmethod
import random

from core.ttp_solution import TTPSolution

class BaseSolver(ABC):
    def __init__(self, max_iterations: int = 1000, on_iteration=None,
                 stagnation_window: int = 100, stagnation_tol: float = 1e-4):
        self.max_iterations = max_iterations
        self.on_iteration = on_iteration
        self.stagnation_window = stagnation_window
        self.stagnation_tol = stagnation_tol
        self._best_score = None
        self._best_solution = None

    @abstractmethod
    def _initialize(self, instance):
        pass

    @abstractmethod
    def _iterate(self, i):
        pass

    @abstractmethod
    def _finalize(self):  # returns SolverResult
        pass

    def solve(self, instance):
        start = time.time()
        self._initialize(instance)
        score_history = []
        for i in range(self.max_iterations):
            self._iterate(i)
            if self.on_iteration:
                self.on_iteration(i, self._best_score)
            if self._best_score is not None:
                score_history.append(self._best_score)
                if len(score_history) >= self.stagnation_window:
                    old = score_history[-self.stagnation_window]
                    if abs(self._best_score - old) < self.stagnation_tol * (abs(old) + 1e-12):
                        break
        result = self._finalize()
        result.runtime = time.time() - start
        return result

    def _greedy_tour(self, start: int = 0) -> list[int]:
        n = self.instance.n
        visited = [False] * n
        visited[start] = True
        tour = [start]
        current = start
        for _ in range(n - 1):
            best_dist = float('inf')
            best_next = -1
            for j in range(n):
                if not visited[j] and self.instance.distances[current][j] < best_dist:
                    best_dist = self.instance.distances[current][j]
                    best_next = j
            visited[best_next] = True
            tour.append(best_next)
            current = best_next
        return tour

    def _greedy_tour_cost(self) -> float:
        tour = self._greedy_tour()
        total = sum(self.instance.distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        total += self.instance.distances[tour[-1]][tour[0]]
        return total
        
    def _greedy_packing(self, tour = None) -> list[int]:
        n = self.instance.n

        if tour is not None:
            remaining = [0.0] * n
            for k in range(n - 2, -1, -1):
                remaining[k] = remaining[k + 1] + self.instance.distances[tour[k]][tour[k + 1]]
            city_to_pos = {city: k for k, city in enumerate(tour)}

            def score(item):
                # in last step, dist=0, avoid div by 0 and take as much as you can  
                dist = remaining[city_to_pos[item.city]] + 1e-9             
                return item.profit / (item.weight * dist)
        else:
            def score(item):
                return item.profit / item.weight

        sorted_items = sorted(self.instance.items, key=score, reverse=True)

        packing = [0] * self.instance.m
        weight = 0.0
        for item in sorted_items:
            if weight + item.weight <= self.instance.capacity:
                packing[item.id] = 1
                weight += item.weight

        return packing


    def _pack_iterative(self, tour: list[int], max_passes: int = None,
                        initial_packing: list[int] = None) -> list[int]:
        """
        Bit-flip local search on packing. Scans all items, applies each improving
        flip immediately, repeats until no improvement.

        max_passes=None          → full convergence.
        max_passes=1             → single pass (MATLS T2=1 style).
        initial_packing=None     → start from empty packing.
        initial_packing=<list>   → warm-start from a given packing (e.g. greedy).
        """
        if initial_packing is not None:
            packing = list(initial_packing)
            current_weight = sum(self.instance.items[k].weight * packing[k]
                                 for k in range(self.instance.m))
        else:
            packing = [0] * self.instance.m
            current_weight = 0.0
        current_score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
        improved = True
        passes = 0
        while improved and (max_passes is None or passes < max_passes):
            improved = False
            passes += 1
            for k in range(self.instance.m):
                w = self.instance.items[k].weight
                if packing[k] == 0 and current_weight + w > self.instance.capacity:
                    continue
                packing[k] ^= 1
                new_score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
                if new_score > current_score:
                    current_weight += w if packing[k] == 1 else -w
                    current_score = new_score
                    improved = True
                else:
                    packing[k] ^= 1  # revert
        return packing

    def _two_opt_full(self, tour: list[int], packing: list[int],
                      max_passes: int = None) -> list[int]:
        n = self.instance.n
        passes = 0
        improved = True
        while improved and (max_passes is None or passes < max_passes):
            improved = False
            passes += 1
            current_score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))

            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    score = self.evaluator.evaluate(TTPSolution(tour=new_tour, packing=packing))
                    if score > current_score:
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break

        return tour

    def _or_opt_full(self, tour: list[int], packing: list[int],
                     max_passes: int = None) -> list[int]:
        n = self.instance.n
        passes = 0
        improved = True
        while improved and (max_passes is None or passes < max_passes):
            improved = False
            passes += 1
            best_score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
            for seg_len in (1, 2):
                for i in range(n - seg_len + 1):
                    segment = tour[i:i + seg_len]
                    remaining = tour[:i] + tour[i + seg_len:]
                    for j in range(len(remaining) + 1):
                        if j == i:
                            continue
                        new_tour = remaining[:j] + segment + remaining[j:]
                        score = self.evaluator.evaluate(TTPSolution(tour=new_tour, packing=packing))
                        if score > best_score:
                            best_score = score
                            tour = new_tour
                            improved = True
        return tour

    @staticmethod
    def _mutate_tour(tour: list[int]) -> list[int]:
        new_tour = tour[:]
        a, b = sorted(random.sample(range(len(tour)), 2))
        new_tour[a:b+1] = new_tour[a:b+1][::-1]
        return new_tour

    @staticmethod
    def _mutate_pack(packing: list[int], instance) -> list[int]:
        new_pack = packing[:]
        idx = random.randrange(instance.m)
        new_pack[idx] ^= 1
        if new_pack[idx] == 1:
            weight = sum(instance.items[k].weight * new_pack[k] for k in range(instance.m))
            if weight > instance.capacity:
                new_pack[idx] = 0
        return new_pack

    @staticmethod
    def _ox_crossover(tour_a: list[int], tour_b: list[int]) -> list[int]:
        n = len(tour_a)
        start, end = sorted(random.sample(range(n), 2))
        segment = set(tour_a[start:end + 1])
        child = [None] * n
        child[start:end + 1] = tour_a[start:end + 1]
        b_iter = (c for c in tour_b if c not in segment)
        for i in list(range(end + 1, n)) + list(range(start)):
            child[i] = next(b_iter)
        return child

    @staticmethod
    def _uniform_crossover(pack_a: list[int], pack_b: list[int]) -> list[int]:
        return [a if random.random() < 0.5 else b for a, b in zip(pack_a, pack_b)]

    @staticmethod
    def _repair_packing(packing: list[int], instance) -> list[int]:
        weight = sum(instance.items[k].weight * packing[k] for k in range(instance.m))
        if weight <= instance.capacity:
            return packing
        new_pack = packing[:]
        taken = sorted(
            (k for k in range(instance.m) if new_pack[k] == 1),
            key=lambda k: instance.items[k].profit / instance.items[k].weight,
        )
        for k in taken:
            if weight <= instance.capacity:
                break
            new_pack[k] = 0
            weight -= instance.items[k].weight
        return new_pack
