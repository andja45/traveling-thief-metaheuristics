import random

from core.ttp_evaluator import TTPEvaluator
from core.ttp_solution import TTPSolution
from solvers.base_solver import BaseSolver
from solvers.solver_result import SolverResult


class GAImproved(BaseSolver):
    
    def __init__(self, pop_size=30, crossover_rate=0.85, mutation_rate=0.15,
                 tournament_size=3, max_iterations=1000,
                 tour_opt_passes=3,pack_opt_passes=3, on_iteration=None):
        super().__init__(max_iterations=max_iterations, on_iteration=on_iteration)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.tour_opt_passes = tour_opt_passes
        self.pack_opt_passes=pack_opt_passes


    @staticmethod
    def _eax(parent_a: list[int], parent_b: list[int], distances) -> list[int]:
        n = len(parent_a)

        # position of each city in tour arrays
        pos_a = [0] * n
        pos_b = [0] * n
        for k, c in enumerate(parent_a):
            pos_a[c] = k
        for k, c in enumerate(parent_b):
            pos_b[c] = k

        def nxt_a(c): return parent_a[(pos_a[c] + 1) % n]
        def nxt_b(c): return parent_b[(pos_b[c] + 1) % n]
        def prv_b(c): return parent_b[(pos_b[c] - 1) % n]

        # find AB-cycles, go in reverse for B to modify successors
        visited = [False] * n
        ab_cycles = []
        for start in range(n):
            if visited[start]:
                continue
            cycle = []
            v = start
            while not visited[v]:
                cycle.append(v)
                visited[v] = True
                v = prv_b(nxt_a(v))
            ab_cycles.append(cycle)

        non_trivial = [c for c in ab_cycles if len(c) > 1]
        if not non_trivial:
            return list(parent_a)   # parents are identical

        e_set = set(random.choice(non_trivial))

        # swap B outgoing edge for cities in e-set
        child_succ = [nxt_b(c) if c in e_set else nxt_a(c) for c in range(n)]

        # make tours
        seen = [False] * n
        subtours = []
        for start in range(n):
            if seen[start]:
                continue
            sub = []
            c = start
            while not seen[c]:
                sub.append(c)
                seen[c] = True
                c = child_succ[c]
            subtours.append(sub)

        if len(subtours) == 1:
            return subtours[0]

        # patch subtours, merge two greedy
        while len(subtours) > 1:
            s1 = subtours.pop(0)
            s2 = subtours.pop(0)
            n1, n2 = len(s1), len(s2)
            best_delta = float('inf')
            best_i, best_j = 0, 0
            for i in range(n1):
                a, a2 = s1[i], s1[(i + 1) % n1]
                for j in range(n2):
                    b, b2 = s2[j], s2[(j + 1) % n2]
                    # swap to a->b2, b->a2
                    delta = (distances[a][b2] + distances[b][a2]
                             - distances[a][a2] - distances[b][b2])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            i, j = best_i, best_j
            merged = s1[:i + 1] + s2[j + 1:] + s2[:j + 1] + s1[i + 1:]
            subtours.insert(0, merged)

        return subtours[0]

    def _tournament(self) -> int:
        candidates = random.sample(range(self.pop_size), self.tournament_size)
        return max(candidates, key=lambda i: self.scores[i])


    def _initialize(self, instance):
        self.instance = instance
        self.evaluator = TTPEvaluator(instance)
        self.convergence = []

        # greedy seed NN + 2 opt and pack iterative
        starts = random.sample(range(instance.n), min(self.pop_size, instance.n))
        while len(starts) < self.pop_size:
            starts.append(random.randrange(instance.n))

        self.population = []
        for s in starts:
            tour = self._greedy_tour(s)
            packing = self._greedy_packing(tour)
            tour = self._two_opt_full(tour, packing, max_passes=self.tour_opt_passes*2)
            packing = self._pack_iterative(tour)
            self.population.append(TTPSolution(tour=tour, packing=packing))

        self.scores = [self.evaluator.evaluate(s) for s in self.population]
        best_idx = max(range(self.pop_size), key=lambda i: self.scores[i])
        self._best_score = self.scores[best_idx]
        self._best_solution = TTPSolution(
            tour=self.population[best_idx].tour[:],
            packing=self.population[best_idx].packing[:],
        )

    def _iterate(self, _):
        # steady state, one offspring per iteration
        p1_idx = self._tournament()
        p2_idx = self._tournament()
        p1 = self.population[p1_idx]
        p2 = self.population[p2_idx]

        # crossover
        if random.random() < self.crossover_rate:
            child_tour = self._eax(p1.tour, p2.tour, self.instance.distances)
        else:
            child_tour = p1.tour[:]

        # mutation
        if random.random() < self.mutation_rate:
            child_tour = self._mutate_tour(child_tour)

        # local search on offspring
        child_tour = self._two_opt_full(child_tour, p1.packing, max_passes=self.tour_opt_passes)
        child_tour = self._or_opt_full(child_tour, p1.packing, max_passes=self.tour_opt_passes)

        # packing greedy and 1 bit-flip pass 
        child_pack = self._greedy_packing(child_tour)
        child_pack = self._pack_iterative(child_tour, max_passes=self.pack_opt_passes, initial_packing=child_pack)
        child = TTPSolution(tour=child_tour, packing=child_pack)
        child_score = self.evaluator.evaluate(child)

        worst_idx = min(range(self.pop_size), key=lambda i: self.scores[i])
        if child_score > self.scores[worst_idx]:
            self.population[worst_idx] = child
            self.scores[worst_idx] = child_score

        if child_score > self._best_score:
            self._best_score = child_score
            self._best_solution = TTPSolution(tour=child_tour[:], packing=child_pack[:])

        self.convergence.append(self._best_score)

    def _finalize(self):
        return SolverResult(
            solution=self._best_solution,
            algo="GA_Improved",
            convergence=self.convergence,
            runtime=0.0,
        )
