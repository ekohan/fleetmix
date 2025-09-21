"""
Testing framework for evaluating improvements to mixed fleet optimization.
Allows A/B testing of changes against problematic instances.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from fleetmix.api import optimize
from fleetmix.config import FleetMixParams
from fleetmix.experiments.alpha_analysis.fleet_templates import (
    make_mixed_fleet,
    make_scv_fleet,
)
from fleetmix.utils.data_processing import load_customer_demand


@dataclass
class TestResult:
    """Result of a single test case."""
    instance: str
    alpha: float
    C: float
    baseline_cost: float
    improved_cost: float
    scv_cost: float
    cost_improvement: float
    cost_improvement_pct: float
    still_loses_to_scv: bool
    baseline_time: float
    improved_time: float
    baseline_gap: float
    improved_gap: float
    additional_metrics: Dict


@dataclass
class TestSummary:
    """Summary of all test results."""
    total_cases: int
    baseline_scv_wins: int
    improved_scv_wins: int
    scv_wins_reduction: float
    mean_cost_improvement: float
    median_cost_improvement: float
    max_cost_improvement: float
    cases_improved: int
    cases_degraded: int
    mean_time_change: float
    detailed_results: List[TestResult]


@dataclass
class HypothesisResult:
    """Result of testing a hypothesis."""
    hypothesis: str
    description: str
    sample_size: int
    baseline_scv_wins: int
    improved_scv_wins: int
    scv_wins_reduction_pct: float
    mean_cost_improvement_pct: float
    success: bool  # True if improvement > threshold
    detailed_results: List[TestResult]
    recommendation: str


class ImprovementTester:
    """Framework for testing improvements to mixed fleet optimization."""
    
    def __init__(self, problematic_cases: List[Tuple[str, float, float]] = None):
        """
        Initialize tester with list of problematic cases.
        
        Args:
            problematic_cases: List of (instance, alpha, C) tuples to test
        """
        self.cases = problematic_cases or []
        self.results: List[TestResult] = []
        self.diagnostic = None  # Will be initialized when needed
        
    def _get_diagnostic(self):
        """Get or create diagnostic instance."""
        if self.diagnostic is None:
            from fleetmix.experiments.alpha_analysis.scv_wins_diagnostic import SCVWinsDiagnostic
            self.diagnostic = SCVWinsDiagnostic()
        return self.diagnostic
    
    def load_hypothesis_cases(self, hypothesis: str, n: int = 15) -> None:
        """
        Load cases for a specific hypothesis.
        
        Args:
            hypothesis: One of H1-H6 hypothesis codes
            n: Number of cases to load
        """
        diagnostic = self._get_diagnostic()
        self.cases = diagnostic.select_hypothesis_sample(hypothesis, n=n)
        print(f"Loaded {len(self.cases)} cases for hypothesis {hypothesis}")
    
    def run_baseline(self, instance: str, alpha: float, C: float) -> Tuple[Dict, float]:
        """Run baseline mixed fleet optimization."""
        demand_path = Path(f"data/demand_profiles/{instance}.csv")
        customers_df = load_customer_demand(str(demand_path))
        
        params = make_mixed_fleet(alpha=alpha, C=C, demand_day=instance, allow_split=True)
        
        start_time = time.time()
        solution = optimize(customers_df, params)
        runtime = time.time() - start_time
        
        result = {
            'cost': solution.total_cost,
            'vehicles': solution.total_vehicles,
            'fixed_cost': solution.total_fixed_cost,
            'variable_cost': solution.total_variable_cost,
            'penalties': solution.total_penalties,
            'gap': solution.optimality_gap or 0.0,
            'solver_status': solution.solver_status,
            'vehicles_used': solution.vehicles_used,
        }
        
        return result, runtime
    
    def run_improved(
        self, 
        instance: str, 
        alpha: float, 
        C: float,
        param_modifier: Optional[Callable[[FleetMixParams], FleetMixParams]] = None
    ) -> Tuple[Dict, float]:
        """
        Run improved mixed fleet optimization.
        
        Args:
            instance: Demand instance name
            alpha: Alpha parameter
            C: C parameter
            param_modifier: Optional function to modify parameters
        """
        demand_path = Path(f"data/demand_profiles/{instance}.csv")
        customers_df = load_customer_demand(str(demand_path))
        
        params = make_mixed_fleet(alpha=alpha, C=C, demand_day=instance, allow_split=True)
        
        # Apply modifications
        if param_modifier:
            params = param_modifier(params)
        
        start_time = time.time()
        solution = optimize(customers_df, params)
        runtime = time.time() - start_time
        
        result = {
            'cost': solution.total_cost,
            'vehicles': solution.total_vehicles,
            'fixed_cost': solution.total_fixed_cost,
            'variable_cost': solution.total_variable_cost,
            'penalties': solution.total_penalties,
            'gap': solution.optimality_gap or 0.0,
            'solver_status': solution.solver_status,
            'vehicles_used': solution.vehicles_used,
        }
        
        return result, runtime
    
    def run_scv_baseline(self, instance: str) -> Dict:
        """Run SCV baseline for comparison."""
        demand_path = Path(f"data/demand_profiles/{instance}.csv")
        customers_df = load_customer_demand(str(demand_path))
        
        params = make_scv_fleet(instance)
        solution = optimize(customers_df, params)
        
        return {
            'cost': solution.total_cost,
            'vehicles': solution.total_vehicles,
            'gap': solution.optimality_gap or 0.0,
        }
    
    def test_improvement(
        self,
        param_modifier: Optional[Callable[[FleetMixParams], FleetMixParams]] = None,
        description: str = "Unnamed improvement"
    ) -> TestSummary:
        """
        Test an improvement across all problematic cases.
        
        Args:
            param_modifier: Function to modify parameters for improved version
            description: Description of the improvement being tested
        """
        print(f"\nTesting improvement: {description}")
        print(f"Running {len(self.cases)} test cases...")
        
        self.results = []
        
        for i, (instance, alpha, C) in enumerate(self.cases, 1):
            print(f"\rProgress: {i}/{len(self.cases)}", end="", flush=True)
            
            # Run baseline
            baseline_result, baseline_time = self.run_baseline(instance, alpha, C)
            
            # Run improved
            improved_result, improved_time = self.run_improved(
                instance, alpha, C, param_modifier
            )
            
            # Get SCV cost
            scv_result = self.run_scv_baseline(instance)
            
            # Calculate improvements
            cost_improvement = baseline_result['cost'] - improved_result['cost']
            cost_improvement_pct = 100.0 * cost_improvement / baseline_result['cost']
            
            # Create test result
            result = TestResult(
                instance=instance,
                alpha=alpha,
                C=C,
                baseline_cost=baseline_result['cost'],
                improved_cost=improved_result['cost'],
                scv_cost=scv_result['cost'],
                cost_improvement=cost_improvement,
                cost_improvement_pct=cost_improvement_pct,
                still_loses_to_scv=improved_result['cost'] > scv_result['cost'],
                baseline_time=baseline_time,
                improved_time=improved_time,
                baseline_gap=baseline_result['gap'],
                improved_gap=improved_result['gap'],
                additional_metrics={
                    'baseline_vehicles': baseline_result['vehicles'],
                    'improved_vehicles': improved_result['vehicles'],
                    'baseline_penalties': baseline_result['penalties'],
                    'improved_penalties': improved_result['penalties'],
                }
            )
            
            self.results.append(result)
        
        print("\nDone!")
        
        # Generate summary
        return self._generate_summary()
    
    def test_hypothesis(
        self,
        hypothesis: str,
        param_modifier: Optional[Callable[[FleetMixParams], FleetMixParams]] = None,
        description: str = None,
        n_samples: int = 15,
        success_threshold: float = 0.2  # 20% reduction in SCV wins
    ) -> HypothesisResult:
        """
        Test a hypothesis-driven improvement.
        
        Args:
            hypothesis: One of H1-H6 hypothesis codes
            param_modifier: Function to modify parameters for improved version
            description: Description of the improvement (auto-generated if None)
            n_samples: Number of instances to test
            success_threshold: Minimum reduction in SCV wins to consider success
            
        Returns:
            HypothesisResult with detailed analysis
        """
        # Load cases for this hypothesis
        self.load_hypothesis_cases(hypothesis, n=n_samples)
        
        # Auto-generate description if not provided
        if description is None:
            hypothesis_descriptions = {
                "H1": "Solver configuration harmonization",
                "H2": "Vehicle configuration optimization",
                "H3": "Initial solution improvement", 
                "H4": "Penalty calculation precision",
                "H5": "Constraint formulation fix",
                "H6": "Cost calculation standardization"
            }
            description = hypothesis_descriptions.get(hypothesis, f"Hypothesis {hypothesis}")
        
        # Run the test
        print(f"\nTesting hypothesis {hypothesis}: {description}")
        print(f"Target: Reduce SCV wins by at least {success_threshold*100:.0f}%")
        
        summary = self.test_improvement(param_modifier, description)
        
        # Calculate hypothesis-specific metrics
        baseline_scv_wins = summary.baseline_scv_wins
        improved_scv_wins = summary.improved_scv_wins
        
        if baseline_scv_wins > 0:
            scv_wins_reduction_pct = (baseline_scv_wins - improved_scv_wins) / baseline_scv_wins
        else:
            scv_wins_reduction_pct = 0.0
        
        # Determine success and recommendation
        success = scv_wins_reduction_pct >= success_threshold
        
        if success:
            if scv_wins_reduction_pct >= 0.5:  # 50% reduction
                recommendation = "STRONGLY RECOMMEND: Significant improvement achieved"
            else:
                recommendation = "RECOMMEND: Meaningful improvement achieved"
        else:
            if scv_wins_reduction_pct >= 0.1:  # 10% reduction
                recommendation = "REFINE: Some improvement shown, worth refining"
            elif summary.mean_cost_improvement > 0:
                recommendation = "INVESTIGATE: Cost improvements but insufficient SCV wins reduction"
            else:
                recommendation = "REJECT: No meaningful improvement"
        
        # Create hypothesis result
        result = HypothesisResult(
            hypothesis=hypothesis,
            description=description,
            sample_size=summary.total_cases,
            baseline_scv_wins=baseline_scv_wins,
            improved_scv_wins=improved_scv_wins,
            scv_wins_reduction_pct=scv_wins_reduction_pct,
            mean_cost_improvement_pct=summary.mean_cost_improvement,
            success=success,
            detailed_results=summary.detailed_results,
            recommendation=recommendation
        )
        
        return result
    
    def _generate_summary(self) -> TestSummary:
        """Generate summary of test results."""
        baseline_scv_wins = sum(1 for r in self.results if r.baseline_cost > r.scv_cost)
        improved_scv_wins = sum(1 for r in self.results if r.still_loses_to_scv)
        
        cost_improvements = [r.cost_improvement_pct for r in self.results]
        time_changes = [r.improved_time - r.baseline_time for r in self.results]
        
        return TestSummary(
            total_cases=len(self.results),
            baseline_scv_wins=baseline_scv_wins,
            improved_scv_wins=improved_scv_wins,
            scv_wins_reduction=baseline_scv_wins - improved_scv_wins,
            mean_cost_improvement=sum(cost_improvements) / len(cost_improvements),
            median_cost_improvement=sorted(cost_improvements)[len(cost_improvements) // 2],
            max_cost_improvement=max(cost_improvements),
            cases_improved=sum(1 for c in cost_improvements if c > 0),
            cases_degraded=sum(1 for c in cost_improvements if c < 0),
            mean_time_change=sum(time_changes) / len(time_changes),
            detailed_results=self.results
        )
    
    def save_results(self, summary: TestSummary, filename: str = None):
        """Save test results to file."""
        if filename is None:
            filename = f"test_results_{int(time.time())}.json"
        
        output_path = Path("improvement_test_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert to serializable format
        results_data = {
            'summary': {
                'total_cases': summary.total_cases,
                'baseline_scv_wins': summary.baseline_scv_wins,
                'improved_scv_wins': summary.improved_scv_wins,
                'scv_wins_reduction': summary.scv_wins_reduction,
                'mean_cost_improvement': summary.mean_cost_improvement,
                'median_cost_improvement': summary.median_cost_improvement,
                'max_cost_improvement': summary.max_cost_improvement,
                'cases_improved': summary.cases_improved,
                'cases_degraded': summary.cases_degraded,
                'mean_time_change': summary.mean_time_change,
            },
            'detailed_results': [
                {
                    'instance': r.instance,
                    'alpha': r.alpha,
                    'C': r.C,
                    'baseline_cost': r.baseline_cost,
                    'improved_cost': r.improved_cost,
                    'scv_cost': r.scv_cost,
                    'cost_improvement': r.cost_improvement,
                    'cost_improvement_pct': r.cost_improvement_pct,
                    'still_loses_to_scv': r.still_loses_to_scv,
                    'baseline_time': r.baseline_time,
                    'improved_time': r.improved_time,
                    'baseline_gap': r.baseline_gap,
                    'improved_gap': r.improved_gap,
                    'additional_metrics': r.additional_metrics,
                }
                for r in summary.detailed_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, summary: TestSummary):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total cases tested: {summary.total_cases}")
        print(f"Baseline SCV wins: {summary.baseline_scv_wins}")
        print(f"Improved SCV wins: {summary.improved_scv_wins}")
        print(f"Reduction in SCV wins: {summary.scv_wins_reduction}")
        print(f"\nCost improvements:")
        print(f"  Mean: {summary.mean_cost_improvement:.2f}%")
        print(f"  Median: {summary.median_cost_improvement:.2f}%")
        print(f"  Max: {summary.max_cost_improvement:.2f}%")
        print(f"  Cases improved: {summary.cases_improved}")
        print(f"  Cases degraded: {summary.cases_degraded}")
        print(f"\nMean time change: {summary.mean_time_change:.2f}s")
        print("="*60)
    
    def print_hypothesis_result(self, result: HypothesisResult):
        """Print hypothesis test result."""
        print("\n" + "="*60)
        print(f"HYPOTHESIS {result.hypothesis} TEST RESULT")
        print("="*60)
        print(f"Description: {result.description}")
        print(f"Sample size: {result.sample_size} instances")
        print(f"\nSCV Wins:")
        print(f"  Baseline: {result.baseline_scv_wins}")
        print(f"  Improved: {result.improved_scv_wins}")
        print(f"  Reduction: {result.scv_wins_reduction_pct:.1%}")
        print(f"\nMean cost improvement: {result.mean_cost_improvement_pct:.2f}%")
        print(f"\nSuccess: {'YES' if result.success else 'NO'}")
        print(f"Recommendation: {result.recommendation}")
        
        # Show top improvements
        if result.detailed_results:
            improvements = sorted(
                result.detailed_results, 
                key=lambda r: r.cost_improvement_pct, 
                reverse=True
            )[:5]
            print("\nTop 5 improvements:")
            for r in improvements:
                print(f"  {r.instance} (α={r.alpha}, C={r.C}): {r.cost_improvement_pct:.1f}%")
        
        print("="*60)
    
    def save_hypothesis_result(self, result: HypothesisResult, filename: str = None):
        """Save hypothesis test result to file."""
        if filename is None:
            filename = f"hypothesis_{result.hypothesis}_{int(time.time())}.json"
        
        output_path = Path("improvement_test_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert to serializable format
        data = {
            'hypothesis': result.hypothesis,
            'description': result.description,
            'sample_size': result.sample_size,
            'baseline_scv_wins': result.baseline_scv_wins,
            'improved_scv_wins': result.improved_scv_wins,
            'scv_wins_reduction_pct': result.scv_wins_reduction_pct,
            'mean_cost_improvement_pct': result.mean_cost_improvement_pct,
            'success': result.success,
            'recommendation': result.recommendation,
            'detailed_results': [
                {
                    'instance': r.instance,
                    'alpha': r.alpha,
                    'C': r.C,
                    'baseline_cost': r.baseline_cost,
                    'improved_cost': r.improved_cost,
                    'scv_cost': r.scv_cost,
                    'cost_improvement': r.cost_improvement,
                    'cost_improvement_pct': r.cost_improvement_pct,
                    'still_loses_to_scv': r.still_loses_to_scv,
                    'baseline_time': r.baseline_time,
                    'improved_time': r.improved_time,
                    'baseline_gap': r.baseline_gap,
                    'improved_gap': r.improved_gap,
                    'additional_metrics': r.additional_metrics,
                }
                for r in result.detailed_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


# Example improvement functions
def increase_time_limit(params: FleetMixParams) -> FleetMixParams:
    """Increase solver time limit."""
    params.runtime.solver_time_limit = 300  # 5 minutes
    return params


def add_scv_priority(params: FleetMixParams) -> FleetMixParams:
    """Add preference for SCV vehicles in mixed fleet."""
    # This would require implementation in the solver
    # For now, just a placeholder
    return params


def improve_initial_solution(params: FleetMixParams) -> FleetMixParams:
    """Use better initial solution generation."""
    # This would require implementation in the solver
    # For now, just a placeholder
    return params


def main():
    """Example usage of hypothesis-based improvement testing."""
    print("=" * 60)
    print("HYPOTHESIS-BASED IMPROVEMENT TESTING")
    print("=" * 60)
    
    # Initialize tester
    tester = ImprovementTester()
    
    # Example: Test H1 (Solver configuration)
    print("\nTesting Hypothesis H1: Solver Configuration Differences")
    print("-" * 60)
    
    # Test with increased time limit
    h1_result = tester.test_hypothesis(
        hypothesis="H1",
        param_modifier=increase_time_limit,
        description="Increased solver time limit to 5 minutes",
        n_samples=15
    )
    
    tester.print_hypothesis_result(h1_result)
    tester.save_hypothesis_result(h1_result)
    
    # Example: Test without any changes to establish baseline
    print("\n\nEstablishing baseline for H3 (Initial Solution)")
    print("-" * 60)
    
    h3_baseline = tester.test_hypothesis(
        hypothesis="H3",
        param_modifier=None,  # No changes
        description="Baseline - no modifications",
        n_samples=20
    )
    
    tester.print_hypothesis_result(h3_baseline)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review results and recommendations")
    print("2. Implement core pipeline changes for successful hypotheses")
    print("3. Run full validation on larger sample if needed")
    print("4. Document changes and update codebase")


if __name__ == "__main__":
    main()
