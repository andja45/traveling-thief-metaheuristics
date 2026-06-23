from itertools import permutations
import time

from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.solver_result import SolverResult


class BruteForceSolver:
    def solve(self, instance) -> SolverResult:
        start = time.time()
        evaluator = TTPEvaluator(instance)
        n, m = instance.n, instance.m

        best_score = float('-inf')
        best_solution = None
        convergence = []

        
        for tour_suffix in permutations(range(1, n)):
            tour = [0] + list(tour_suffix)

            for mask in range(1 << m):
                packing = [(mask >> k) & 1 for k in range(m)]
                weight = sum(instance.items[k].weight * packing[k] for k in range(m))
                if weight > instance.capacity:
                    continue

                score = evaluator.evaluate(TTPSolution(tour=tour, packing=packing))
                if score > best_score:
                    best_score = score
                    best_solution = TTPSolution(tour=list(tour), packing=list(packing))
                convergence.append(best_score)

        return SolverResult(
            solution=best_solution,
            algo="BruteForce",
            convergence=convergence,
            runtime=time.time() - start,
        )
