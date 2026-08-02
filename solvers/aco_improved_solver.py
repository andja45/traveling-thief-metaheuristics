import random
import time
from core.ttp_solution import TTPSolution
from core.ttp_evaluator import TTPEvaluator
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult

class ACOImproved(BaseSolver):
    def __init__(self, n_ants=None, max_iterations=500, on_iteration=None, alpha=1.0, beta=2.0, rho=0.02, p_best=0.05,
                 stagnation_limit=100, rho_scale=2.0):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.p_best = p_best
        self.stagnation_limit = stagnation_limit # iterations without improvement before tau reset
        self.rho_scale = rho_scale # how much rho can grow at max stagnation (2.0 = up to 3x base rate)

    def _update_tau_bounds(self, best_tour_cost: float):
        self.tau_max = 1.0 / (self.rho * best_tour_cost)

        p = self.p_best ** (1.0 / self.instance.n)
        avg = self.instance.n / 2
        self.tau_min = self.tau_max * (1 - p) / ((avg - 1) * p)

    def _build_tour(self) -> list[int]:
        n = self.instance.n

        start = 0
        tour = [start]
        visited = [False] * n
        visited[start] = True

        for _ in range(n - 1):
            current = tour[-1]
            candidates = []
            weights = []

            for j in range(n):
                if not visited[j]:
                    w = (self.tau[current][j] ** self.alpha) * (self.eta[current][j] ** self.beta)
                    candidates.append(j)
                    weights.append(w)

            next_city = random.choices(candidates, weights=weights, k=1)[0]

            tour.append(next_city)
            visited[next_city] = True

        return tour

    def _tour_cost(self, tour: list[int]) -> float:
        n = self.instance.n
        return sum(
            self.instance.distances[tour[k]][tour[(k + 1) % n]]
            for k in range(n)
        )

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        if self.n_ants is None:
            self.n_ants = instance.n
        self._convergence = []
        self.tau_history = []
        self._start_time = time.time()
        self._stagnation_count = 0

        self._update_tau_bounds(self._random_tour_cost())

        self.tau = [[self.tau_max] * instance.n for _ in range(instance.n)]
        self.eta = [[0.0] * instance.n for _ in range(instance.n)]
        for i in range(instance.n):
            for j in range(instance.n):
                if i != j:
                    self.eta[i][j] = (1.0 / instance.distances[i][j])

    def _iterate(self, iteration):
        best_iter_score = None
        best_iter_tour = None
        best_iter_packing = None

        for _ in range(self.n_ants):
            tour = self._build_tour()
            packing = self._greedy_packing(tour)
            score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))

            if best_iter_score is None or score > best_iter_score:
                best_iter_score = score
                best_iter_tour = tour
                best_iter_packing = packing

        best_iter_tour = self._two_opt_full(best_iter_tour, best_iter_packing, max_passes=1)
        best_iter_packing = self._pack_iterative(best_iter_tour, max_passes=3, initial_packing=best_iter_packing)
        best_iter_score = self.evaluator.evaluate(TTPSolution(tour=best_iter_tour, packing=best_iter_packing))

        improved = self._best_score is None or best_iter_score > self._best_score
        if improved:
            self._update_tau_bounds(self._tour_cost(best_iter_tour))
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        if self._stagnation_count >= self.stagnation_limit:
            self.tau = [[self.tau_max] * self.instance.n for _ in range(self.instance.n)]
            self._stagnation_count = 0

        self._record(best_iter_score, TTPSolution(tour=best_iter_tour, packing=best_iter_packing))

        # rho grows with stagnation - more evaporation when stuck, more exploration
        stagnation_ratio = min(1.0, self._stagnation_count / self.stagnation_limit)
        rho = self.rho * (1.0 + self.rho_scale * stagnation_ratio)

        for a in range(self.instance.n):
            for b in range(self.instance.n):
                self.tau[a][b] *= (1.0 - rho)

        # deposit source switches to global best once stagnation passes 50%, not on a fixed iteration
        deposit_tour = self._best_solution.tour if stagnation_ratio > 0.5 else best_iter_tour
        deposit = 1.0 / self._tour_cost(deposit_tour)

        for k in range(self.instance.n):
            a = deposit_tour[k]
            b = deposit_tour[(k + 1) % self.instance.n]
            self.tau[a][b] += deposit

        for a in range(self.instance.n):
            for b in range(self.instance.n):
                if self.tau[a][b] > self.tau_max:
                    self.tau[a][b] = self.tau_max
                elif self.tau[a][b] < self.tau_min:
                    self.tau[a][b] = self.tau_min

        if improved:
            self.tau_history.append([row[:] for row in self.tau])

    def _finalize(self) -> SolverResult:
        return SolverResult(
              solution = self._best_solution,
              algo = "ACO_Improved",
              convergence = self._convergence,
              runtime = time.time() - self._start_time,
          )
