from dataclasses import dataclass
from core.ttp_solution import TTPSolution


@dataclass
class SolverResult:
    solution: TTPSolution
    algo: str
    convergence: list[float]  # best score per iteration
    runtime: float
