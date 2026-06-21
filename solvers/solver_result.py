from dataclasses import dataclass
from core.ttp_solution import TTPSolution

@dataclass
class SolverResult:
    solution: TTPSolution 
    algo: str 
    convergence: list[float] # list of best scores per iteration
    runtime: float 