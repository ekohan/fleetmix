"""
Benchmark runner script for single-compartment VRP solutions.
"""
import os
import sys
from pathlib import Path
import time
import argparse
import numpy as np
import logging
from warnings import warn

from fleetmix.config.parameters import Parameters
from fleetmix.utils.logging import setup_logging, ProgressTracker, Colors, Symbols
from fleetmix.utils.data_processing import load_customer_demand
from fleetmix.benchmarking.solvers.vrp_solver import VRPSolver
from fleetmix.core_types import BenchmarkType, VRPSolution

def parse_benchmark_args():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(
        description='Run VRP benchmarking'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='Time limit for VRP solver in seconds'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--benchmark-type',
        type=str,
        choices=['single_compartment', 'multi_compartment'],
        default='single_compartment',
        help='Type of benchmark to run'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display detailed information about the benchmark tool'
    )
    return parser.parse_args()

def print_solution_details(solution: VRPSolution) -> None:
    """Print solution details."""
    print(f"\nℹ️ VRP Solution Summary:")
    print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{solution.total_cost:,.2f}{Colors.RESET}")
    print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{solution.total_distance:.1f} km{Colors.RESET}")
    print(f"{Colors.BLUE}→ Vehicles Used: {Colors.BOLD}{solution.num_vehicles}{Colors.RESET}")
    
    if solution.vehicle_utilization:
        avg_utilization = sum(float(u) for u in solution.vehicle_utilization) / len(solution.vehicle_utilization)
        print(f"{Colors.BLUE}→ Avg Vehicle Utilization: {Colors.BOLD}{avg_utilization:.1f}%{Colors.RESET}")
    
    print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{solution.execution_time:.1f}s{Colors.RESET}")

def main():
    """Run VRP benchmarking."""
    # Add deprecation warning
    warn("Direct script execution is deprecated. Use 'fleetmix benchmark' instead", FutureWarning)
    
    args = parse_benchmark_args()
    
    if args.info:
        print(f"\n{Colors.BOLD}VRP Benchmark Tool{Colors.RESET}")
        print(f"\n{Colors.BLUE}Description:{Colors.RESET}")
        print("Evaluates Multi-Compartment Vehicle (MCV) Fleet Size and Mix (FSM)")
        print("model against classic single-compartment Vehicle Routing Problem (VRP)")
        print("solutions for comparative analysis.")
        
        print(f"\n{Colors.BLUE}Benchmark Types:{Colors.RESET}")
        print(f"{Colors.CYAN}→ single_compartment:{Colors.RESET} Upper bound using dedicated vehicles per product")
        print(f"{Colors.CYAN}→ multi_compartment:{Colors.RESET} Lower bound assuming perfect compartment flexibility")
        
        print(f"\n{Colors.BLUE}Key Features:{Colors.RESET}")
        print("• Parallel processing for product-specific VRPs")
        print("• PyVRP solver with genetic algorithm optimization")
        print("• Detailed solution metrics and analysis")
        print("• Support for standard CVRP benchmark instances")
        
        print(f"\n{Colors.BLUE}Example Usage:{Colors.RESET}")
        print("python run_benchmark.py --benchmark-type single_compartment --time-limit 300")
        return
    
    setup_logging()
    
    # Load parameters
    params = Parameters.from_yaml(args.config)
    
    # Define benchmark steps
    steps = [
        'Load Data',
        'Run VRP Solver',
        'Save Results'
    ]
    
    progress = ProgressTracker(steps)
    
    # Step 1: Load customer data
    customers = load_customer_demand(params.demand_file)
    progress.advance(
        f"Loaded {Colors.BOLD}{len(customers)}{Colors.RESET} customers"
    )
    
    # Step 2: Run VRP solver
    benchmark_type = BenchmarkType(args.benchmark_type)
    vrp_solver = VRPSolver(
        customers=customers,
        params=params,
        time_limit=args.time_limit,
        benchmark_type=benchmark_type
    )
    solutions = vrp_solver.solve(verbose=args.verbose)
    
    # Calculate totals across all product solutions
    total_cost = sum(sol.total_cost for sol in solutions.values())
    total_vehicles = sum(sol.num_vehicles for sol in solutions.values())
    
    progress.advance(
        f"Completed VRP: {Colors.BOLD}${total_cost:,.2f}{Colors.RESET} "
        f"total cost, {Colors.BOLD}{total_vehicles}{Colors.RESET} vehicles"
    )
    
    # Print detailed results for each product
    for product, solution in solutions.items():
        print(f"\n{Colors.BOLD}{product} Product Results:{Colors.RESET}")
        print_solution_details(solution)
    
    # Step 3: Save results
    from fleetmix.utils.save_results import save_benchmark_results
    save_benchmark_results(
        solutions=solutions,
        parameters=params,
        benchmark_type=benchmark_type,
        format='excel'  # Could add this as a CLI argument if needed
    )
    progress.advance("Results saved")
    
    progress.close()

if __name__ == "__main__":
    main() 