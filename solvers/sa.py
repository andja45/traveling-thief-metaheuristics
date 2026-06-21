import math
import random

from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult


class SA(BaseSolver):
    def __init__(self, max_iterations=50000, T_init=500.0, cooling=0.9995, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.T_init = T_init
        self.cooling = cooling

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        self.T = self.T_init
        self.convergence = []

        tour = self._greedy_tour()
        packing = self._greedy_packing(tour)

        self._current_solution = TTPSolution(tour=tour, packing=packing)
        self._current_score = self.evaluator.evaluate(self._current_solution)
        self._current_weight = sum(instance.items[k].weight * packing[k] for k in range(instance.m))
        self._best_solution = TTPSolution(tour=tour[:], packing=packing[:])
        self._best_score = self._current_score

    def _iterate(self, _):
        tour = self._current_solution.tour
        packing = self._current_solution.packing
        move = random.random()

        if move < 0.4:  # 2-opt
            a, b = sorted(random.sample(range(self.instance.n), 2))
            new_tour = tour[:a] + tour[a:b + 1][::-1] + tour[b + 1:]
            new_pack = packing[:]
            new_weight = self._current_weight
        elif move < 0.7:  # relocate city
            i = random.randrange(self.instance.n)
            j = random.randrange(self.instance.n)
            new_tour = list(tour)
            city = new_tour.pop(i)
            new_tour.insert(j, city)
            new_pack = packing[:]
            new_weight = self._current_weight
        else:
            new_tour = tour
            new_pack = packing[:]
            new_weight = self._current_weight

        flip = random.random()
        if flip < 0.4:  # bit-flip (independent, can compound with tour move)
            idx = random.randrange(self.instance.m)
            new_pack[idx] ^= 1
            item_w = self.instance.items[idx].weight
            new_weight = (new_weight + item_w) if new_pack[idx] else (new_weight - item_w)
            if new_weight > self.instance.capacity:
                new_pack[idx] ^= 1
                new_weight -= item_w
                

        candidate = TTPSolution(tour=new_tour, packing=new_pack)
        new_score = self.evaluator.evaluate(candidate)
        delta = new_score - self._current_score

        if delta > 0 or random.random() < math.exp(delta / max(self.T, 1e-10)):
            self._current_solution = candidate
            self._current_score = new_score
            self._current_weight = new_weight
            if new_score > self._best_score:
                self._best_score = new_score
                self._best_solution = TTPSolution(tour=list(new_tour), packing=list(new_pack))

        self.convergence.append(self._best_score)
        self.T *= self.cooling

    def _finalize(self):
        return SolverResult(solution=self._best_solution,algo="SA",
            convergence=self.convergence,runtime=0.0)
