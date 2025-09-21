#!/usr/bin/env python3
"""
Quick test script for Hypothesis H1: Solver Configuration Differences

This script tests whether harmonizing solver parameters between SCV and mixed fleet
runs can reduce the cases where SCV baseline outperforms mixed fleet.
"""

from pathlib import Path
from fleetmix.config import FleetMixParams
from fleetmix.experiments.alpha_analysis.improvement_tester import ImprovementTester
from fleetmix.experiments.alpha_analysis.scv_wins_diagnostic import SCVWinsDiagnostic


def increase_solver_time_limit(params: FleetMixParams) -> FleetMixParams:
    """H1 Fix: Increase solver time limit to ensure convergence."""
    params.runtime.solver_time_limit = 300  # 5 minutes
    return params


def harmonize_solver_params(params: FleetMixParams) -> FleetMixParams:
    """H1 Fix: Ensure consistent solver parameters."""
    # Increase time limit
    params.runtime.solver_time_limit = 300  # 5 minutes
    
    # Ensure consistent MIP gap
    if hasattr(params.runtime, 'mip_gap'):
        params.runtime.mip_gap = 0.001  # 0.1% gap
    
    # Ensure sufficient iterations
    if hasattr(params.runtime, 'max_iterations'):
        params.runtime.max_iterations = 10000
        
    return params


def main():
    """Test H1 hypothesis with targeted instances."""
    print("=" * 70)
    print("TESTING HYPOTHESIS H1: Solver Configuration Differences")
    print("=" * 70)
    
    # First, analyze which instances are most affected
    print("\n1. Analyzing instances most likely affected by solver issues...")
    diagnostic = SCVWinsDiagnostic()
    h1_analysis = diagnostic.analyze_hypothesis("H1")
    
    print(f"\nH1 Analysis:")
    print(f"- Sample size: {h1_analysis['sample_size']} instances")
    print(f"- Mean cost penalty: {h1_analysis['mean_cost_penalty_pct']:.1f}%")
    print(f"- Max cost penalty: {h1_analysis['max_cost_penalty_pct']:.1f}%")
    
    if 'mean_optimality_gap' in h1_analysis['characteristics']:
        print(f"- Mean optimality gap: {h1_analysis['characteristics']['mean_optimality_gap']:.3f}")
    
    # Test the hypothesis
    print("\n2. Testing improvement: Harmonized solver parameters")
    print("-" * 70)
    
    tester = ImprovementTester()
    
    # Test 1: Just increase time limit
    print("\nTest 1: Increased time limit only")
    result1 = tester.test_hypothesis(
        hypothesis="H1",
        param_modifier=increase_solver_time_limit,
        description="Increased solver time limit to 5 minutes",
        n_samples=15
    )
    
    tester.print_hypothesis_result(result1)
    tester.save_hypothesis_result(result1, "h1_time_limit_only.json")
    
    # Test 2: Full harmonization (if first test shows promise)
    if result1.scv_wins_reduction_pct >= 0.1:  # At least 10% improvement
        print("\n\nTest 2: Full solver parameter harmonization")
        result2 = tester.test_hypothesis(
            hypothesis="H1", 
            param_modifier=harmonize_solver_params,
            description="Full solver parameter harmonization",
            n_samples=15
        )
        
        tester.print_hypothesis_result(result2)
        tester.save_hypothesis_result(result2, "h1_full_harmonization.json")
    else:
        print("\n\nSkipping Test 2: Insufficient improvement from time limit increase")
    
    # Summary
    print("\n" + "=" * 70)
    print("H1 TEST SUMMARY")
    print("=" * 70)
    
    if result1.success:
        print("✓ H1 hypothesis CONFIRMED: Solver configuration affects results")
        print(f"  - SCV wins reduced by {result1.scv_wins_reduction_pct:.1%}")
        print(f"  - Mean cost improvement: {result1.mean_cost_improvement_pct:.1f}%")
        print("\nRECOMMENDATION: Implement solver harmonization in core pipeline")
        print("  - Update fleetmix/solver/base_solver.py")
        print("  - Ensure consistent parameters across all problem types")
    else:
        print("✗ H1 hypothesis NOT CONFIRMED")
        print("  - Solver configuration changes show minimal impact")
        print("  - Proceed to test other hypotheses")
    
    print("\nDetailed results saved to improvement_test_results/")


if __name__ == "__main__":
    main()
