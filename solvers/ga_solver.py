import random

from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult


class GA(BaseSolver):
    def __init__(self, pop_size=50, crossover_rate=0.8, mutation_rate=0.1,
                 tournament_size=3, max_iterations=500, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        self.convergence = []

        greedy_tour = self._greedy_tour()
        self.population = [TTPSolution(tour=greedy_tour, packing=self._greedy_packing(greedy_tour))]

        for _ in range(self.pop_size - 1):
            tour = random.sample(range(instance.n), instance.n)
            self.population.append(TTPSolution(tour=tour, packing=self._greedy_packing(tour)))

        self.scores = [self.evaluator.evaluate(s) for s in self.population]
        best_idx = max(range(self.pop_size), key=lambda i: self.scores[i])
        self._best_score = self.scores[best_idx]
        self._best_solution = TTPSolution(
            tour=list(self.population[best_idx].tour),
            packing=list(self.population[best_idx].packing),
        )

    def _tournament(self) -> TTPSolution:
        candidates = random.sample(range(self.pop_size), self.tournament_size)
        return self.population[max(candidates, key=lambda i: self.scores[i])]

    def _iterate(self, _):
        elite_idx = max(range(self.pop_size), key=lambda i: self.scores[i])
        elite = TTPSolution(
            tour=list(self.population[elite_idx].tour),
            packing=list(self.population[elite_idx].packing),
        )
        elite_score = self.scores[elite_idx]

        offspring = []
        for _ in range(self.pop_size):
            pa = self._tournament()
            pb = self._tournament()

            if random.random() < self.crossover_rate:
                child_tour = self._ox_crossover(pa.tour, pb.tour)
                child_pack = self._repair_packing(self._uniform_crossover(pa.packing, pb.packing), self.instance)
            else:
                child_tour = list(pa.tour)
                child_pack = list(pa.packing)

            if random.random() < self.mutation_rate:
                child_tour = self._mutate_tour(child_tour)
            if random.random() < self.mutation_rate:
                child_pack = self._mutate_pack(child_pack, self.instance)

            offspring.append(TTPSolution(tour=child_tour, packing=child_pack))

        off_scores = [self.evaluator.evaluate(s) for s in offspring]

        worst_idx = min(range(self.pop_size), key=lambda i: off_scores[i])
        offspring[worst_idx] = elite
        off_scores[worst_idx] = elite_score

        self.population = offspring
        self.scores = off_scores

        best_idx = max(range(self.pop_size), key=lambda i: self.scores[i])
        if self.scores[best_idx] > self._best_score:
            self._best_score = self.scores[best_idx]
            self._best_solution = TTPSolution(
                tour=list(self.population[best_idx].tour),
                packing=list(self.population[best_idx].packing),
            )

        self.convergence.append(self._best_score)

    def _finalize(self):
        return SolverResult(
            solution=self._best_solution,
            algo="GA",
            convergence=self.convergence,
            runtime=0.0,
        )
