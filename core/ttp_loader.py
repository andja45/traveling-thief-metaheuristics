from core.ttp_instance import TTPInstance, Item
import math

def _distance(x1, y1, x2, y2, edge_weight_type: str) -> float:
      d = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
      if edge_weight_type == "CEIL_2D":
          return math.ceil(d)
      return d

def load_instance(filepath: str) -> TTPInstance:
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("DIMENSION"):
                n = int(line.split(":")[1].strip())
            elif line.startswith("NUMBER OF ITEMS"):
                m = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY OF KNAPSACK"):
                capacity = int(line.split(":")[1].strip())
            elif line.startswith("MIN SPEED"):
                v_min = float(line.split(":")[1].strip())
            elif line.startswith("MAX SPEED"):
                v_max = float(line.split(":")[1].strip())
            elif line.startswith("RENTING RATIO"):
                renting_rate = float(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                break

        cities = []
        for _ in range(n):
            line = f.readline().strip()
            _, x, y = line.split()
            cities.append((float(x), float(y)))

        f.readline()  # skip "ITEMS SECTION"

        items = []
        for _ in range(m):
            line = f.readline().strip()
            id, profit, weight, city = line.split()
            items.append(Item(id=int(id)-1, profit=float(profit), weight=float(weight), city=int(city)-1)) # convert to 0-based index

        items_by_city = {}
        for item in items:
            items_by_city.setdefault(item.city, []).append(item)
        
        distances = [[_distance(cities[i][0], cities[i][1], cities[j][0], cities[j][1], edge_weight_type) for j in range(n)] for i in range(n)]

        return TTPInstance(n=n, m=m, cities=cities, items=items, items_by_city=items_by_city, capacity=capacity, v_min=v_min, v_max=v_max, renting_rate=renting_rate, distances=distances)

