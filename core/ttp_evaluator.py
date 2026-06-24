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

    def _normalize_tour(self, tour: list[int]) -> list[int]:
        idx = tour.index(0)
        return tour[idx:] + tour[:idx]

    def evaluate(self, solution: TTPSolution) -> float:
        weight = 0.0
        total_time = 0.0
        profit = 0.0
        tour = self._normalize_tour(solution.tour)
        current_city = tour[0]

        for city in tour[1:]:
            for item in self.instance.items_by_city.get(current_city, []):
                if solution.packing[item.id] == 1:
                    weight += item.weight
                    profit += item.profit

            distance = self.instance.distances[current_city][city]
            speed = self._speed(weight)
            total_time += distance / speed

            current_city = city

        # last city, pick up items, then return to city 0
        for item in self.instance.items_by_city.get(current_city, []):
            if solution.packing[item.id] == 1:
                weight += item.weight
                profit += item.profit

        if current_city != 0:
            distance = self.instance.distances[current_city][0]
            total_time += distance / self._speed(weight)

        return self._objective(profit, total_time)

    def evaluate_traced(self, solution: TTPSolution) -> tuple[float, list[dict]]:
        weight = 0.0
        total_time = 0.0
        profit = 0.0
        tour = self._normalize_tour(solution.tour)
        current_city = tour[0]

        trace = []

        for city in tour[1:]:
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

        if current_city != 0:
            return_dist = self.instance.distances[current_city][0]
            return_speed = self._speed(weight)
            return_time = return_dist / return_speed
            total_time += return_time
        else:
            return_dist = None
            return_speed = None
            return_time = None

        trace.append({
            "city": current_city,
            "next_city": 0 if current_city != 0 else None,
            "items_picked": items_picked,
            "weight": weight,
            "speed": return_speed,
            "distance": return_dist,
            "time": return_time,
            "profit_so_far": profit,
            "objective_so_far": self._objective(profit, total_time)
        })

        return self._objective(profit, total_time), trace
