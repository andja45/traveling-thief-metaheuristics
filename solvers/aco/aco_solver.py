import random
import time
from core.ttp_solution import TTPSolution
from core.ttp_evaluator import TTPEvaluator
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult

class ACOSolver(BaseSolver):
    def __init__(self, n_ants=None, max_iterations=500, on_iteration=None, alpha=1.0, beta=2.0, rho=0.02, p_best=0.05):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.n_ants = n_ants # based on the instance
        self.alpha = alpha # pheromone influence
        self.beta = beta # heuristic influence
        self.rho = rho # evaporation rate
        self.p_best = p_best # prob of constructing best tour (taking a step towards it)

    def _update_tau_bounds(self, best_tour_cost: float):
        self.tau_max = 1.0 / (self.rho * best_tour_cost) # pheromone added 1/greedy_cost = pheromone removed rho * tau - stable here

        p = self.p_best ** (1.0 / self.instance.n) # probabilty of taking the same road - p^n = pbest
        avg = self.instance.n / 2 # estimate of unvisited cities per step
        self.tau_min = self.tau_max * (1 - p) / ((avg - 1) * p) # p = tau_min / (tau_min + (avg-1) * tau_max) prob of picking underexplored edge

    def _build_tour(self) -> list[int]:
        n = self.instance.n

        # each ant starts from a different city
        start = random.randint(0, n - 1)
        tour = [start]
        visited = [False] * n
        visited[start] = True

        for _ in range(n - 1):
            current = tour[-1]
            candidates = []
            weights = []

            for j in range(n):
                if not visited[j]:
                    # probability balancing pheromone and heuristic ifluence
                    w = (self.tau[current][j] ** self.alpha) * (self.eta[current][j] ** self.beta)
                    candidates.append(j)
                    weights.append(w)

            # random selection weighted by probability
            next_city = random.choices(candidates, weights=weights, k=1)[0]

            tour.append(next_city)
            visited[next_city] = True

        return tour

    def _tour_cost(self, tour: list[int]) -> float:
        # sum of edge distances along the closed tour (last city → first city included)
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
        self._start_time = time.time()
        self._stagnation_count = 0  # iterations without improvement

        self._update_tau_bounds(self._greedy_tour_cost())

        self.tau = [[self.tau_max] * instance.n for _ in range(instance.n)] # make all edges equally attractive (initialize to tau_max)
        # eta is heuristic attractiveness of edge - we use item density aware eta 
        self.eta = [[0.0] * instance.n for _ in range(instance.n)]
        for i in range(instance.n):
            for j in range(instance.n):
                if i != j:
                    self.eta[i][j] = (1.0 / instance.distances[i][j])

    def _iterate(self, i):
        best_iter_score = None
        best_iter_tour = None
        best_iter_packing = None

        for _ in range(self.n_ants):
            tour = self._build_tour()
            packing = self._greedy_packing(tour)
            score = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))

            # update local best
            if best_iter_score is None or score > best_iter_score:
                best_iter_score = score
                best_iter_tour = tour
                best_iter_packing = packing

        # update global best
        if self._best_score is None or best_iter_score > self._best_score:
            self._best_score = best_iter_score
            self._best_solution = TTPSolution(tour=best_iter_tour, packing=best_iter_packing)
            self._update_tau_bounds(self._tour_cost(best_iter_tour))
            self._stagnation_count = 0  # improvement found, reset counter
        else:
            self._stagnation_count += 1

        # of no improvement for 100 iterations make all edges equally attractive (tau_max init)
        if self._stagnation_count >= 100:
            self.tau = [[self.tau_max] * self.instance.n for _ in range(self.instance.n)]
            self._stagnation_count = 0

        self._convergence.append(self._best_score)

        # evaporate pheromone on all edges (give less weight to old paths - chance to explore more)
        for i in range(self.instance.n):
            for j in range(self.instance.n):
                self.tau[i][j] *= (1.0 - self.rho)

        # in MMAS only the best ant deposits pheromone (reinforce best path)
        best_tour = self._best_solution.tour
        deposit = 1.0 / self._tour_cost(best_tour) # shortest the tour higher the pheromone

        for k in range(self.instance.n):
            i = best_tour[k]
            j = best_tour[(k + 1) % self.instance.n]
            self.tau[i][j] += deposit # tau is not symetric

        for i in range(self.instance.n):
            for j in range(self.instance.n):
                if self.tau[i][j] > self.tau_max:
                    self.tau[i][j] = self.tau_max # prevents one path from dominating
                elif self.tau[i][j] < self.tau_min:
                    self.tau[i][j] = self.tau_min # prevents paths from being ignored

    def _finalize(self) -> SolverResult:
        return SolverResult(
              solution = self._best_solution,
              algo = "ACO",
              convergence = self._convergence,
              runtime = time.time() - self._start_time,
          )

