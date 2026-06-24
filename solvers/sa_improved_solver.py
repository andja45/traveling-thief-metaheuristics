import math
import random
import time

from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult


class SAImproved(BaseSolver):
    def __init__(self, max_iterations=50000, T_init=None, cooling=0.9995,
                 T_abs=1.0, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.T_init = T_init
        self.cooling = cooling
        self.T_abs = T_abs

    def _calibrate_temperature(self, tour, packing):
        current_score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
        current_weight = sum(self.instance.items[k].weight * packing[k]
                             for k in range(self.instance.m))
        trial = packing[:]
        deltas = []
        for _ in range(200):
            k = random.randrange(self.instance.m)
            w = self.instance.items[k].weight
            if trial[k] == 0 and current_weight + w > self.instance.capacity:
                continue
            trial[k] ^= 1
            score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=trial))
            delta = score - current_score
            if delta < 0:
                deltas.append(abs(delta))
            trial[k] ^= 1
        if deltas:
            return -sum(deltas) / len(deltas) / math.log(0.8)
        return 500.0

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        self.convergence = []

        tour = self._greedy_tour()
        packing = self._greedy_packing(tour)
        tour = self._two_opt_full(tour, packing)
        tour = self._or_opt_full(tour, packing)
        packing = self._pack_iterative(tour)

        self._tour = tour
        self._packing = packing
        self._score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
        self._weight = sum(instance.items[k].weight * packing[k] for k in range(instance.m))
        self._best_solution = TTPSolution(tour=tour[:], packing=packing[:])
        self._best_score = self._score
        self.T = self.T_init if self.T_init is not None else self._calibrate_temperature(tour, packing)

    def solve(self, instance):
        start = time.time()
        self._initialize(instance)
        i = 0
        while self.T > self.T_abs and i < self.max_iterations:
            self._iterate(i)
            if self.on_iteration:
                self.on_iteration(i, self._best_score)
            i += 1
        result = self._finalize()
        result.runtime = time.time() - start
        return result

    def _iterate(self, _):
        for _ in range(self.instance.n // 2):
            if random.random() < 0.5:
                a, b = sorted(random.sample(range(self.instance.n), 2))
                new_tour = self._tour[:a] + self._tour[a:b + 1][::-1] + self._tour[b + 1:]
            else:
                i = random.randrange(self.instance.n)
                j = random.randrange(self.instance.n)
                if i == j:
                    continue
                new_tour = list(self._tour)
                city = new_tour.pop(i)
                new_tour.insert(j, city)
            new_score = self.evaluator.evaluate(TTPSolution(tour=new_tour, packing=self._packing))
            if new_score > self._score:
                self._tour = new_tour
                self._score = new_score

        for _ in range(self.instance.m // 2):
            k = random.randrange(self.instance.m)
            w = self.instance.items[k].weight
            if self._packing[k] == 0 and self._weight + w > self.instance.capacity:
                continue
            self._packing[k] ^= 1
            new_score = self.evaluator.evaluate(TTPSolution(tour=self._tour, packing=self._packing))
            gain = new_score - self._score
            if gain > 0 or random.random() < math.exp(gain / max(self.T, 1e-10)):
                self._weight += w if self._packing[k] == 1 else -w
                self._score = new_score
            else:
                self._packing[k] ^= 1

        if self._score > self._best_score:
            self._best_score = self._score
            self._best_solution = TTPSolution(tour=self._tour[:], packing=self._packing[:])

        self.convergence.append(self._best_score)
        self.T *= self.cooling

    def _finalize(self):
        return SolverResult(
            solution=self._best_solution,
            algo="SA_Improved",
            convergence=self.convergence,
            runtime=0.0,
        )
