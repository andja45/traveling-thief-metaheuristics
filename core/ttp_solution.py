from dataclasses import dataclass

@dataclass
class TTPSolution:
    tour: list[int] # tour of cities
    packing: list[int] # binary vector, packing[k]=1 means item k is taken (corresponds to items list in instance)