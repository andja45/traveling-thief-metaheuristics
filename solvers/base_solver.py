import time
from abc import ABC, abstractmethod
import random

class BaseSolver(ABC):
    def __init__(self, max_iterations: int = 1000, on_iteration=None):
        self.max_iterations = max_iterations
        self.on_iteration = on_iteration
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
        for i in range(self.max_iterations):
            self._iterate(i)
            if self.on_iteration:
                self.on_iteration(i, self._best_score)
        result = self._finalize()
        result.runtime = time.time() - start
        return result

    def _greedy_tour(self) -> list[int]:
        n = self.instance.n
        visited = [False] * n
        current = 0
        visited[0] = True
        tour = [0]
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


    def _mutate_tour(tour: list[int]) -> list[int]:
        new_tour = tour[:]
        a, b = sorted(random.sample(range(len(tour)), 2))
        new_tour[a:b+1] = new_tour[a:b+1][::-1]
        return new_tour

    def _mutate_pack(packing: list[int], instance) -> list[int]:
        new_pack = packing[:]
        idx = random.randrange(instance.m)
        new_pack[idx] ^= 1
        if new_pack[idx] == 1:
            weight = sum(instance.items[k].weight * new_pack[k] for k in range(instance.m))
            if weight > instance.capacity:
                new_pack[idx] = 0
        return new_pack


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


    def _uniform_crossover(pack_a: list[int], pack_b: list[int]) -> list[int]:
        return [a if random.random() < 0.5 else b for a, b in zip(pack_a, pack_b)]


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
