# Traveling Thief Problem - Metaheuristic Comparison

Comparative study of four metaheuristics (ACO, GWO, GA, SA) and their improved variants on the Traveling Thief Problem - a bi-component NP-hard problem where TSP routing and 0/1 Knapsack selection are coupled through a shared weight-dependent speed objective.

**Tech focus:**  
Python 3 · NumPy · Matplotlib · Jupyter

**Algorithm focus:**  
Ant Colony Optimization · Grey Wolf Optimizer · Genetic Algorithm (EAX) · Simulated Annealing · Nature-Inspired Optimization · Combinatorial Optimization

---

## Problem

The Traveling Thief Problem couples two NP-hard subproblems with a non-linear interaction:

- **Tour** - visit all `n` cities exactly once (TSP)  
- **Packing** - select items from cities to carry home (0/1 Knapsack)

The coupling: picked-up items increase knapsack weight, which slows the thief, increasing renting cost. Solving each subproblem independently yields suboptimal TTP solutions. NP-hard (Bonyadi et al., 2013).

**Objective (maximize):**

```
f(tour, packing) = profit − renting_rate × total_time

speed(w) = v_max − (w / W) × (v_max − v_min)

total_time = Σ distance(city_i → city_{i+1}) / speed(weight_at_city_i)
```

Items are collected at each city en route, so weight (and therefore speed) is tour-order-dependent.

---

## Solvers

All solvers share a `BaseSolver` base class. Subclasses implement `_initialize()`, `_iterate()`, and `_finalize()`; stagnation-based early stopping is built into the solve loop.

| Shared method | Description |
|---|---|
| `_greedy_packing` | Tour-aware greedy packing, scores items by `profit / (weight × remaining_distance)` |
| `_pack_iterative` | Multi-pass bit-flip local search on the packing vector |
| `_two_opt_full` | 2-opt that evaluates the full TTP objective, not just tour distance |
| `_or_opt_full` | OR-opt with segments of length 1 and 2 |

### Brute Force

Exhaustive search over all `(n-1)! × 2^m` combinations. Guarantees the optimal solution; only feasible on tiny instances (tiny7: 720 × 64 = 46,080 evaluations). Used as the reference ceiling.

### S5 - Greedy Baseline

Deterministic, no iterations. Nearest-neighbor tour → greedy packing by `profit / (weight × remaining_distance)` → iterative bit-flip to local optimum. Serves as a lower reference point.

### ACO - Ant Colony Optimization

MMAS (Max-Min Ant System) variant with adaptive pheromone bounds, stagnation recovery, and a deposit schedule that transitions from exploration to exploitation mid-run.

| Feature | Design decision |
|---|---|
| **τ bounds** | τ_max, τ_min derived from best tour cost each improvement - prevents edge dominance and path starvation |
| **Deposit schedule** | Iteration-best for first 50% of iterations, global-best for second half - broadens early exploration, sharpens late convergence |
| **Stagnation reset** | After 100 iterations without improvement, τ reset uniformly to τ_max |
| **η (heuristic)** | Inverse edge distance `1/d(i,j)` - standard attractiveness, independent of packing |
| **Local search** | 1-pass 2-opt on tour + 3-pass iterative bit-flip on packing, applied to iteration-best ant only |

### GWO - Grey Wolf Optimizer

Discrete adaptation of GWO for combined permutation (tour) + binary (packing) solution spaces.

| Feature | Design decision |
|---|---|
| **Pack seeding** | Top 10% of wolves initialized with perturbed greedy tour (random 2-opt reversal) + greedy packing |
| **`a` schedule** | Linearly decays `2 → 0` over iterations - high exploration early, convergence pressure late |
| **Tour movement** | Swap-sequence decomposition: minimal `(i, j)` swap list from wolf to each guide; `1 − \|A\|` fraction applied; `\|A\| ≥ 1` triggers random 2-opt instead |
| **Packing movement** | Bit-copy probability = `max(0, 1 − \|A\|)` toward each guide independently |
| **Majority vote** | Final packing = majority of α/β/δ guide packings per bit; capacity-repair removes lowest-value items if overloaded |

### GA - Genetic Algorithm

| Feature | Design decision |
|---|---|
| **Selection** | Tournament selection (k=3), population of 50 |
| **Tour crossover** | Order crossover (OX) - preserves relative city order from both parents |
| **Packing crossover** | Uniform crossover - each bit independently inherited from either parent |
| **Elitism** | Best individual carried to next generation unchanged |
| **Mutation** | 2-opt reversal for tour; single bit-flip for packing; capacity repair if violated |

### GA (Improved) - Edge Assembly Crossover

| Feature | Design decision |
|---|---|
| **Tour crossover** | EAX (Edge Assembly Crossover) - detects AB-cycles between parent edge sets, merges subtours by minimum distance delta |
| **Initialization** | Every individual seeded from NN tour (different start city) + 2-opt + iterative packing |
| **Local search** | 2-opt + OR-opt applied to every offspring before evaluation |
| **Packing** | Greedy init on offspring tour + multi-pass iterative bit-flip refinement |
| **Replacement** | Steady-state (population of 30): offspring replaces worst only if strictly better |

### SA - Simulated Annealing

| Feature | Design decision |
|---|---|
| **Move types** | 2-opt reversal (p=0.4), city reinsertion (p=0.3), packing-only step (p=0.3); bit-flip applied independently at p=0.4 |
| **Acceptance** | Metropolis criterion: accept worsening move with `exp(Δ/T)` |
| **Cooling** | Geometric: `T ← T × 0.9995` |
| **Capacity** | Bit-flip rejected at move time if it violates capacity - no repair needed |

### SA (Improved) - Adaptive Simulated Annealing

| Feature | Design decision |
|---|---|
| **T₀ calibration** | Sample 200 random bit-flip Δ values from initial solution; set T₀ so `P(accept worsening) ≈ 0.8` - scales automatically per instance |
| **Warm start** | Greedy tour → 2-opt → OR-opt → iterative packing before first iteration |
| **Batch moves** | `n/2` tour proposals (greedy accept) + `m/2` packing flips (SA acceptance) per cooling step |

---

## Failure Modes and Recovery

**Pheromone stagnation (ACO):** Trails converge prematurely, locking the colony into a suboptimal route. After 100 iterations without score improvement, pheromone levels reset to τ_max.

**Constraint violation (GA, GA Improved):** OX crossover preserves tour validity but packing crossover can violate knapsack capacity. Capacity repair removes items in ascending `profit/weight` order until the constraint is satisfied.

**High variance (GWO):** Discrete tour operators struggle to capture fine TTP structure. GWO shows the widest spread across runs (best 110.40, worst 50.54 on tiny7). The discrete adaptation loses the smooth gradients GWO relies on in continuous search spaces.

**Cold-start inefficiency:** SA Improved auto-calibrates T₀ from sampled deltas rather than guessing; GA Improved seeds every individual with a greedy + local-search solution rather than random permutations.

---

## Results

**tiny7** (n=7, m=6), 3 runs. Brute force optimal = **115.23**.

| Algorithm | Best | Mean | Worst | Time (s) |
|---|---|---|---|---|
| Brute Force | 115.23 | n/a | n/a | 0.25 |
| SA Improved | 115.23 | 115.23 | 115.23 | 0.42 |
| ACO (MMAS) | 115.23 | 115.23 | 115.23 | 0.09 |
| SA | 115.23 | 111.53 | 108.95 | 0.11 |
| GA | 110.40 | 109.91 | 108.95 | 0.30 |
| GWO | 110.40 | 90.44 | 50.54 | 0.03 |
| GA Improved | 101.70 | 101.70 | 101.70 | 0.09 |
| S5 | 65.78 | 65.78 | 65.78 | 0.00 |

SA Improved and ACO hit the optimal on every run. GWO has high variance; the discrete operators struggle to capture fine TTP structure. GA Improved underperforms basic GA on tiny instances - EAX overhead and steady-state replacement favour larger instances.

**eil51** (n=51, m=50, bounded-strongly-corr), 3 runs.

| Algorithm | Best | Mean | Worst | Time (s) |
|---|---|---|---|---|
| GA Improved | 4269.21 | 4008.98 | 3806.57 | 41.45 |
| SA Improved | 4000.40 | 3891.81 | 3763.49 | 6.79 |
| ACO (MMAS) | 3819.45 | 3668.63 | 3582.68 | 10.07 |
| GA | 3377.68 | 2954.05 | 2667.50 | 0.71 |
| SA | 2939.51 | 2845.65 | 2797.78 | 0.19 |
| GWO | 2867.72 | 2775.44 | 2640.54 | 0.87 |
| S5 | 2575.12 | 2575.12 | 2575.12 | 0.00 |

GA Improved is the clear winner on larger instances - EAX crossover and local search on every offspring pay off at scale. SA Improved is the best quality/runtime tradeoff (4000 best in under 7s). GWO and basic GA fall significantly behind.

Full results across all instances and algorithms: [`results/all.csv`](results/all.csv)  
Convergence plots, per-instance comparisons, and algorithm analysis: [`docs/notebook.ipynb`](docs/notebook.ipynb)  
Project documentation: [`docs/document.pdf`](docs/document.pdf) · Presentation: [`docs/presentation.pdf`](docs/presentation.pdf)

---

## Benchmark Instances

| Instance | Cities | Items | Correlation |
|---|---|---|---|
| tiny_n5 | 5 | 4 | - |
| tiny_n7 | 7 | 6 | - |
| tiny_n9 | 9 | 8 | - |
| eil51 | 51 | 50 | uncorrelated |
| eil51 | 51 | 50 | bounded-strongly-corr |
| pr76 | 76 | 75 | uncorrelated |
| pr76 | 76 | 75 | bounded-strongly-corr |
| rat195 | 195 | 194 | uncorrelated |
| rat195 | 195 | 194 | bounded-strongly-corr |

---

## Architecture

```
core/
  ttp_instance.py       # cities, items, distances, capacity, renting rate
  ttp_loader.py         # .ttp file parser
  ttp_evaluator.py      # objective function; evaluate_traced for animation
  ttp_solution.py       # (tour, packing) container

solvers/
  base_solver.py        # shared: greedy tour/packing, 2-opt, OR-opt,
                        #         OX crossover, iterative bit-flip, capacity repair
  aco_solver.py
  gwo_solver.py
  ga_solver.py
  ga_improved_solver.py
  sa_solver.py
  sa_improved_solver.py
  s5_solver.py          # greedy construction baseline

visuals/
  convergence.py        # convergence curve generation

docs/
  notebook.ipynb        # full analysis: score tables, convergence plots, comparison
  document.pdf / .tex
  presentation.pdf / .tex

benchmarks/             # .ttp instance files
results/                # all.csv, convergence_raw.png
```

---

## Run

```bash
pip install -r requirements.txt
python main.py
```

Full analysis with plots and comparisons:
```bash
cd docs && jupyter notebook notebook.ipynb
```

---

## References

- Bonyadi, Michalewicz, Barone. *The Travelling Thief Problem*. IEEE CEC 2013.
- Polyakovskiy et al. *A Comprehensive Benchmark Set for TTP*. GECCO 2014.
- Stützle & Hoos. *MAX-MIN Ant System*. FGCS 2000.
- Mirjalili et al. *Grey Wolf Optimizer*. Advances in Engineering Software 2014.
- Kirkpatrick et al. *Optimization by Simulated Annealing*. Science 1983.

---

## Authors

[Andjela Spasic](https://github.com/andja45) · [Matija Radulović](https://github.com/MatijaRadulovic)
