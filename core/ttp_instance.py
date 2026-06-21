from dataclasses import dataclass, field

@dataclass
class Item:
    id: int # item index
    city: int # city index
    profit: float # item profit
    weight: float # item weight

@dataclass
class TTPInstance:
    n: int # num of cities
    m: int # num of items
    cities: list[tuple[float, float]] # cities x,y coord
    items: list[Item] 
    capacity: float # backpack capacity
    v_min: float # speed when backpack is full
    v_max: float # speed when backpack is empty
    renting_rate: float # cost per unit for backpack use
    items_by_city: dict[int, list[Item]] = field(default_factory=dict) 
    distances: list[list[float]] = field(default_factory=list) # euclidean i,j distance
