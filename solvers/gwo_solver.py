import random
import time
from core.ttp_solution import TTPSolution
from core.ttp_evaluator import TTPEvaluator
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult

class GWOSolver(BaseSolver):
    def __init__(self, n_wolves=None, max_iterations=500, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.n_wolves = n_wolves  

    def _initialize(self, instance):
        self.instance  = instance
        self.evaluator = TTPEvaluator(instance)
        self._start_time = time.time()
        self.scores_history = []  # snapshot of all wolf scores per iteration, used for wolf animation

        if self.n_wolves is None:
            self.n_wolves = instance.n

        # start with random wolves 
        self.wolves = [
            TTPSolution(tour=self._random_tour(), packing=self._random_pack())
            for _ in range(self.n_wolves)
        ]

        # seed top 10% with near-greedy solutions 
        greedy_tour = self._greedy_tour()
        for i in range(self.n_wolves // 10):
            tour = greedy_tour[:]
            a, b = sorted(random.sample(range(instance.n), 2))
            tour[a:b+1] = tour[a:b+1][::-1]  # 2-opt reversal
            packing = self._greedy_packing(tour)
            self.wolves[i] = TTPSolution(tour=tour, packing=packing)

        self._rank_wolves()

    def _random_tour(self) -> list[int]:
        suffix = list(range(1, self.instance.n))
        random.shuffle(suffix)
        return [0] + suffix

    def _random_pack(self) -> list[int]:
        # random bits then repair so we never start over capacity
        packing = [random.randint(0, 1) for _ in range(self.instance.m)]
        return BaseSolver._repair_packing(packing, self.instance)

    def _rank_wolves(self):
        # evaluate all wolves, sort best first, keep scores for _iterate
        scores = [self.evaluator.evaluate(w) for w in self.wolves]
        paired = sorted(zip(scores, self.wolves), key=lambda x: x[0], reverse=True)
        self.wolves = [w for _, w in paired]
        self._scores = [s for s, _ in paired]

    def _swap_sequence(self, from_tour, to_tour):
        # builds the minimal list of (i,j) swaps that transforms from_tour into to_tour
        swaps = []
        temp = from_tour[:]
        for i in range(len(temp)):
            if temp[i] != to_tour[i]:
                j = temp.index(to_tour[i], i) # find where the target city currently sits
                swaps.append((i, j))
                temp[i], temp[j] = temp[j], temp[i]
        return swaps

    def _apply_partial_swaps(self, tour, guide_tour, a):
        r1 = random.random()
        A_mag = abs(2 * a * r1 - a)
        if A_mag >= 1:
            result = tour[:]
            x, y = sorted(random.sample(range(len(result)), 2))
            result[x:y+1] = result[x:y+1][::-1]
            return result
        swaps   = self._swap_sequence(tour, guide_tour)
        n_apply = max(1, int((1 - A_mag) * len(swaps))) if swaps else 0
        result  = tour[:]
        for idx in range(n_apply):
            pi, pj = swaps[idx]
            result[pi], result[pj] = result[pj], result[pi]
        return result

    def _update_packing(self, wolf_packing, guide_packing, a):
        # copy differing bits from guide with probability proportional to attraction strength
        r1 = random.random()
        A_mag = abs(2 * a * r1 - a)
        copy_prob = max(0.0, 1 - A_mag)  # 1 = copy everything, 0 = copy nothing
        result = wolf_packing[:]
        for k in range(len(result)):
            if wolf_packing[k] != guide_packing[k] and random.random() < copy_prob:
                result[k] = guide_packing[k]
        return result

    def _move_wolf(self, wolf, alpha, beta, delta, a):
        # apply partial swaps toward each of the three guides sequentially
        # each guide gets an independent A draw so the combined move is balanced
        new_tour = wolf.tour[:]
        new_tour = self._apply_partial_swaps(new_tour, alpha.tour, a)
        new_tour = self._apply_partial_swaps(new_tour, beta.tour,  a)
        new_tour = self._apply_partial_swaps(new_tour, delta.tour, a)

        # packing is majority vote - bit is 1 if >= 2 of 3 guides say 1
        p1 = self._update_packing(wolf.packing, alpha.packing, a)
        p2 = self._update_packing(wolf.packing, beta.packing,  a)
        p3 = self._update_packing(wolf.packing, delta.packing, a)
        new_packing = [
            1 if (p1[k] + p2[k] + p3[k]) >= 2 else 0
            for k in range(self.instance.m)
        ]
        # majority vote can push over capacity - repair before returning
        new_packing = BaseSolver._repair_packing(new_packing, self.instance)

        return TTPSolution(tour=new_tour, packing=new_packing)

    def _iterate(self, i):
        alpha = self.wolves[0] 
        beta  = self.wolves[1]  
        delta = self.wolves[2] 

        a = 2.0 - 2.0 * (i / self.max_iterations)

        for k in range(3, self.n_wolves):
            self.wolves[k] = self._move_wolf(self.wolves[k], alpha, beta, delta, a)

        self._rank_wolves()

        # snapshot wolf scores for animation
        self.scores_history.append(self._scores[:])

        # alpha is now the best wolf after re-ranking
        self._record(self._scores[0], self.wolves[0])

    def _finalize(self) -> SolverResult:
        return SolverResult(
            solution=self._best_solution,
            algo="GWO",
            convergence=self._convergence,
            runtime=time.time() - self._start_time,
        )
