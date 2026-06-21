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