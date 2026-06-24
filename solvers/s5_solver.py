from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult


class S5Solver(BaseSolver):

    def __init__(self, on_iteration=None):
        super().__init__(max_iterations=1, on_iteration=on_iteration)

    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        self._convergence = []

        tour = self._greedy_tour(start=0)

        packing = self._greedy_packing(tour)
        score_greedy = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
        self._convergence.append(score_greedy)

        packing = BaseSolver._pack_iterative(self, tour, initial_packing=packing)
        score_packed = self.evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
        self._convergence.append(score_packed)

        self._best_score = score_packed
        self._best_solution = TTPSolution(tour=tour, packing=packing)

    def _iterate(self, i):
        pass

    def _finalize(self):
        return SolverResult(
            solution=self._best_solution,
            algo="S5",
            convergence=self._convergence,
            runtime=0.0,
        )
