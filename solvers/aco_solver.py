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
        self.p_best = p_best # prob of reconstructing the best tour 

    def _update_tau_bounds(self, best_tour_cost: float):
        self.tau_max = 1.0 / (self.rho * best_tour_cost) # pheromone added 1/greedy_cost = pheromone removed rho * tau - stable here

        p = self.p_best ** (1.0 / self.instance.n) # probabilty of taking the same road - p^n = pbest
        avg = self.instance.n / 2 # estimate of unvisited cities per step
        self.tau_min = self.tau_max * (1 - p) / ((avg - 1) * p) # p = tau_max / (tau_max + (avg-1) * tau_min) prob of picking best edge against (avg-1) worse ones 
        # ^ what value should tau_min have so tau_max edge gets picked

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
        self.tau_history  = []  # snapshot of tau after each iteration, used for pheromone animation
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

    def _iterate(self, iteration):
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

        # 2-opt on tour, then iterative bit-flip packing, all applied to iteration-best only
        best_iter_tour = self._two_opt_full(best_iter_tour, best_iter_packing, max_passes=1)
        best_iter_packing = self._pack_iterative(best_iter_tour, max_passes=3, initial_packing=best_iter_packing)
        best_iter_score = self.evaluator.evaluate(TTPSolution(tour=best_iter_tour, packing=best_iter_packing))
        
        improved = self._best_score is None or best_iter_score > self._best_score
        if improved:
            self._update_tau_bounds(self._tour_cost(best_iter_tour))
            self._stagnation_count = 0  # improvement found, reset counter
        else:
            self._stagnation_count += 1

        # of no improvement for 100 iterations make all edges equally attractive (tau_max init)
        if self._stagnation_count >= 100:
            self.tau = [[self.tau_max] * self.instance.n for _ in range(self.instance.n)]
            self._stagnation_count = 0

        self._record(best_iter_score, TTPSolution(tour=best_iter_tour, packing=best_iter_packing))

        # evaporate pheromone on all edges (give less weight to old paths - chance to explore more)
        for a in range(self.instance.n):
            for b in range(self.instance.n):
                self.tau[a][b] *= (1.0 - self.rho)

        # in MMAS only the best ant deposits pheromone (reinforce best path)
        # best_tour = self._best_solution.tour
        # deposit = 1.0 / self._tour_cost(best_tour) # shortest the tour higher the pheromone

        # early iterations: deposit from iteration-best (explore broadly)
        # late iterations: deposit from global best (converge on best known)
        deposit_tour = best_iter_tour if iteration < self.max_iterations // 2 else self._best_solution.tour
        deposit = 1.0 / self._tour_cost(deposit_tour)

        for k in range(self.instance.n):
            a = deposit_tour[k]
            b = deposit_tour[(k + 1) % self.instance.n]
            self.tau[a][b] += deposit # tau is not symetric

        for a in range(self.instance.n):
            for b in range(self.instance.n):
                if self.tau[a][b] > self.tau_max:
                    self.tau[a][b] = self.tau_max # prevents one path from dominating
                elif self.tau[a][b] < self.tau_min:
                    self.tau[a][b] = self.tau_min # prevents paths from being ignored

        # snapshot tau only when a new best is found 
        if improved:
            self.tau_history.append([row[:] for row in self.tau])

    def _finalize(self) -> SolverResult:
        return SolverResult(
              solution = self._best_solution,
              algo = "ACO",
              convergence = self._convergence,
              runtime = time.time() - self._start_time,
          )

