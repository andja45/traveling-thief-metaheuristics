from core.ttp_loader import load_instance

inst = load_instance("benchmarks/eil51/eil51_n50_uncorr_01.ttp")

class TestHeader:
    def test_city_count(self):
        assert inst.n == 51

    def test_item_count(self):
        assert inst.m == 50

    def test_capacity(self):
        assert inst.capacity == 2226

    def test_speeds(self):
        assert inst.v_min == 0.1
        assert inst.v_max == 1.0

    def test_renting_rate(self):
        assert inst.renting_rate == 7.19

class TestCities:
    def test_count(self):
        assert len(inst.cities) == inst.n

    def test_first_city(self):
        assert inst.cities[0] == (37.0, 52.0)

    def test_second_city(self):
        assert inst.cities[1] == (49.0, 49.0)

class TestItems:
    def test_count(self):
        assert len(inst.items) == inst.m

    def test_first_item_fields(self):
        item = inst.items[0]
        assert item.id == 0
        assert item.profit == 119.0
        assert item.weight == 1.0
        assert item.city == 1  # city 2 in file -> 0-based

    def test_items_by_city_lookup(self):
        assert inst.items[0] in inst.items_by_city[1]

class TestDistances:
    def test_matrix_size(self):
        assert len(inst.distances) == inst.n
        assert len(inst.distances[0]) == inst.n

    def test_self_distance_is_zero(self):
        assert inst.distances[0][0] == 0.0

    def test_symmetric(self):
        assert inst.distances[0][1] == inst.distances[1][0]
