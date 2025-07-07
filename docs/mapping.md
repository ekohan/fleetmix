# Code ↔ Paper Mapping

<!-- TODO: check this -->

This table helps readers of *Designing Multi-Compartment Vehicle Fleets for Last-Mile Food Distribution Systems* navigate between the
research methodology and its open-source implementation.

| Code Module / Script | Purpose in Codebase | Paper Section / Figure |
|----------------------|---------------------|------------------------|
| `src/utils/vehicle_configurations.py` | Enumerate all feasible vehicle types × compartment sets | §4.1 Generate Vehicle Configurations |
| `src/clustering.py` | Build capacity- & time-feasible customer clusters | §4.2 Generate Feasible Customer Clusters (Fig. 1, Algorithm 1) |
| `src/utils/route_time.py` | Route-time estimation heuristics (Legacy, BHH, TSP) | §4.2, footnote on clustering time checks |
| `src/fsm_optimizer.py` | Mixed-integer programme for fleet size-and-mix | §4.3 Fleet Size and Mix Optimisation |
| `src/post_optimization.py` | Iterative merge improvement phase | §4.4 Improvement Phase |
| `src/main.py` | CLI pipeline that chains clustering → MILP → post-opt | §5, §6 Computational Results & Case Study |
| `src/benchmarking/run_benchmark.py` | Batch runner for CVRP → FSM adaptation tests across multiple variants | §5 Effectiveness of Decomposition Approach |
| `src/benchmarking/run_all_mcvrp.py` | Batch runner to reproduce Henke & Hübner (2015) MCVRP benchmark results | §5 Effectiveness of Decomposition Approach |
| `src/benchmarking/vrp_solver.py` | VRP solver interface (PyVRP) for lower/upper bounds in benchmarking | §5 Effectiveness of Decomposition Approach |
| `tests/` | Unit & integration tests ensuring algorithmic correctness | – |

**How to use this page**

1. Find the methodological concept you care about in the right-hand column.
2. Jump to the corresponding Python file to inspect or extend the implementation.
3. Cross-link back to the paper to understand assumptions and notation.

> The docstrings in each module contain only a brief citation (e.g. “See §4.2”).  This mapping page
> gives the full picture without cluttering IDE tool-tips for industry users.

