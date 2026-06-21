import random
import time
from core.ttp_solution import TTPSolution
from core.ttp_evaluator import TTPEvaluator
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult

class GWOSolver(BaseSolver):
    def __init__(self, n_wolves = None, max_iterations=500, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.n_wolves = n_wolves 
        pass

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance) 
        self._convergence = []
        self._start_time = time.time()

        if self.n_wolves is None:
            self.n_wolves = instance.n

        self.wolves = [TTPSolution(tour=self._random_tour(), packing=self._random_pack()) for _ in range(self.n_wolves)]
        
        for i in range(self.n_wolves//10):
            tour = self._mutate_tour(self._greedy_tour())
            packing = self._repair_packing(self._mutate_pack(self._greedy_packing(tour)))

            self.wolves[i] = TTPSolution(tour=tour, packing=packing)
        
        self._rank_wolves()

    def _random_tour(self) -> list[int]:
        tour = list([i for i in range(self.instance.n)])
        random.shuffle(tour)
        return tour
    
    def _random_pack(self) -> list[int]:
        return [random.randint(0, 1) for _ in range(self.instance.m)]

    def _rank_wolves(self):
      scores = [    self.evaluator.evaluate(w) for w in self.wolves]
      paired = sorted(zip(scores, self.wolves),key=lambda x: x[0], reverse=True)
      self.wolves = [w for _, w in paired]


    def _iterate(self, i):
        best_iter_score = None
        best_iter_tour = None
        best_iter_packing = None

        self.alpha=self.wolves[0]
        self.beta=self.wolves[1]
        self.delta=self.wolves[2]

        a = 2 - 2 * (i / self.max_iterations)  
        for i in range(3, self.n_wolves):
            self.wolves[i] = self._move_wolf(self.wolves[i], self.alpha, self.beta, self.delta, a)
    
        self._rank_wolves()
    
        # update local best
        score = self.evaluator.evaluate(self.wolves[0])
        if best_iter_score is None or score > best_iter_score:
            best_iter_score = score
            best_iter_tour = self.wolves[0].tour
            best_iter_packing = self.wolves[0].packing

        # update global best
        if self._best_score is None or best_iter_score > self._best_score:
            self._best_score = best_iter_score
            self._best_solution = TTPSolution(tour=best_iter_tour, packing=best_iter_packing)

        self._convergence.append(self._best_score)

    def _finalize(self) -> SolverResult:
        return SolverResult(
              solution = self._best_solution,
              algo = "GWO",
              convergence = self._convergence,
              runtime = time.time() - self._start_time,
          )

