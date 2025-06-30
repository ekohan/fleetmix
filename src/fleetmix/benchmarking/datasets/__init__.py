"""Benchmark datasets for evaluating fleet design optimization algorithms.

This package contains two sets of standard Vehicle Routing Problem (VRP) benchmark instances:

1. MCVRP (Multi-Compartment Vehicle Routing Problem) instances from Henke et al. (2015):
   - 150 instances with 10 customers and 3 instances with 50 customers
   - Customer demand distributed across three product types
   - Homogeneous fleet with three compartments per vehicle
   - Instances vary by number of product types requested per customer (1, 2, or 3)
   - Vehicle capacity of 1,000 units across all instances

2. CVRP (Capacitated Vehicle Routing Problem) instances from Uchoa et al. (2017):
   - Originally single-product instances adapted for multi-product scenarios
   - Adapted using four strategies: Split, Scaled, Combined, and Spatial Differentiation
   - Various topologies and demand patterns to test robustness

References:
- Henke, T., Speranza, M. G., & Wäscher, G. (2015). The multi-compartment vehicle routing
  problem with flexible compartment sizes. European Journal of Operational Research, 246(3):730-743.
- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017).
  New benchmark instances for the Capacitated Vehicle Routing Problem.
  European Journal of Operational Research, 257(3), 845-858.
"""

# No public exports - datasets are accessed through parsers
__all__: list[str] = []
