from abc import ABC, abstractmethod

class BaseSolver(ABC):
    # subclasses can override with their own value
    def __init__(self, max_iterations: int = 1000, on_iteration = None):
      self.max_iterations = max_iterations
      self.on_iteration = on_iteration
      # updated in _iterate() and populate SolverResult
      self._best_score = None # collected in convergence score
      self._best_solution = None # best TTPSolution so far

    @abstractmethod
    def _initialize(self, instance):
        pass

    @abstractmethod
    def _iterate(self, i):
        pass

    @abstractmethod
    def _finalize(self): # returns SolverResult
        pass

    def solve(self, instance):
        self._initialize(instance)
        for i in range(self.max_iterations):
            self._iterate(i) # first update then report
            if self.on_iteration:
                self.on_iteration(i, self._best_score) # track progress
        return self._finalize()
    
    def _greedy_tour_cost(self) -> float:
        n = self.instance.n

        visited = [False] * n

        # start from city 0 
        current = 0
        visited[0] = True

        total = 0.0

        # visit n-1 remaining cities 
        for _ in range(n - 1):
            best_dist = float('inf') 
            best_next = -1

            for j in range(n):
                if not visited[j] and self.instance.distances[current][j] < best_dist:
                    best_dist = self.instance.distances[current][j]
                    best_next = j # nearest unvisited city

            # move to it and mark visited
            total += best_dist
            visited[best_next] = True
            current = best_next

        # return to starting city
        total += self.instance.distances[current][0]

        return total
        
    def _greedy_packing(self, tour: None) -> list[int]:
        n = self.instance.n

        if tour is not None:
            # remaining distance from each position to end of tour
            remaining = [0.0] * n
            for k in range(n - 2, -1, -1):
                remaining[k] = remaining[k + 1] + self.instance.distances[tour[k]][tour[k + 1]]

            def score(item, pos):
                dist = remaining[pos] + 1e-9  # in last step, dist=0, avoid div by 0 and take as much as you can
                return item.profit / (item.weight * dist)
        else:
            def score(item, pos):
                return item.profit / item.weight

        packing = [0] * self.instance.m
        weight  = 0.0
        city_to_pos = {city: k for k, city in enumerate(tour)}
        
        sorted_items = sorted(self.instance.items,
                            key=lambda item: score(item, city_to_pos[item.city]),
                            reverse=True)

        packing = [0] * self.instance.m  
        weight = 0.0
  
        # pack sorted items until capacity is reached
        for item in sorted_items:
            if weight + item.weight <= self.instance.capacity:
                packing[item.id] = 1  
                weight += item.weight
  
        return packing