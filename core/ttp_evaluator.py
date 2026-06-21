from core.ttp_instance import TTPInstance
from core.ttp_solution import TTPSolution

class TTPEvaluator:
    def __init__(self, instance):
        self.instance = instance

    # calculate speed based on current backpack weight
    def _speed(self, weight: float) -> float:
        return self.instance.v_max - weight / self.instance.capacity * (self.instance.v_max - self.instance.v_min)

    # ttp objective function to maximize
    def _objective(self, profit: float, total_time: float) -> float:
        return profit - self.instance.renting_rate * total_time

    def evaluate(self, solution: TTPSolution) -> float:
        weight = 0.0
        total_time = 0.0
        profit = 0.0
        current_city = solution.tour[0]

        for city in solution.tour[1:]:
            for item in self.instance.items_by_city.get(current_city, []):
                if solution.packing[item.id] == 1:
                    weight += item.weight
                    profit += item.profit

            distance = self.instance.distances[current_city][city]
            speed = self._speed(weight)
            total_time += distance / speed

            current_city = city

        # last city - pickup only
        for item in self.instance.items_by_city.get(current_city, []):
            if solution.packing[item.id] == 1:
                weight += item.weight
                profit += item.profit

        return self._objective(profit, total_time)

    def evaluate_traced(self, solution: TTPSolution) -> tuple[float, list[dict]]:
        weight = 0.0
        total_time = 0.0
        profit = 0.0
        current_city = solution.tour[0]
        
        trace = []

        for city in solution.tour[1:]:
            items_picked = []
            for item in self.instance.items_by_city.get(current_city, []):
                if solution.packing[item.id] == 1:
                    weight += item.weight
                    profit += item.profit
                    items_picked.append(item.id)

            distance = self.instance.distances[current_city][city]
            speed = self._speed(weight)
            time = distance / speed
            total_time += time

            trace.append({
                "city": current_city,
                "next_city": city,
                "items_picked": items_picked,
                "weight": weight,
                "speed": speed,
                "distance": distance,
                "time": time,
                "profit_so_far": profit,
                "objective_so_far": self._objective(profit, total_time)
            })

            current_city = city

        items_picked = []
        for item in self.instance.items_by_city.get(current_city, []):
            if solution.packing[item.id] == 1:
                weight += item.weight
                profit += item.profit
                items_picked.append(item.id)

        trace.append({
            "city": current_city,
            "next_city": None,
            "items_picked": items_picked,
            "weight": weight,
            "speed": None,
            "distance": None,
            "time": None,
            "profit_so_far": profit,
            "objective_so_far": self._objective(profit, total_time)
        })

        return self._objective(profit, total_time), trace
