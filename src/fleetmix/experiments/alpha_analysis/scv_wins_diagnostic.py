"""
Diagnostic tool for analyzing cases where SCV baseline outperforms mixed fleet.
Extracts problematic instances and provides detailed comparisons.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from fleetmix.api import optimize
from fleetmix.experiments.alpha_analysis.fleet_templates import (
    make_mixed_fleet,
    make_scv_fleet,
)
from fleetmix.utils.data_processing import load_customer_demand


class SCVWinsDiagnostic:
    """Analyze cases where SCV baseline beats mixed fleet."""
    
    def __init__(self, summary_path: Path = None):
        """Initialize diagnostic tool with summary data."""
        if summary_path is None:
            summary_path = Path("src/fleetmix/experiments/alpha_analysis/results/summary_mixed.parquet")
        
        self.df = pd.read_parquet(summary_path)
        self.scv_wins = self._identify_scv_wins()
        
    def _identify_scv_wins(self) -> pd.DataFrame:
        """Extract cases where SCV baseline beats mixed fleet."""
        # Get SCV baseline costs
        scv_baseline = self.df[self.df['fleet_type'] == 'SCV_BASE'][
            ['instance', 'total_cost']
        ].rename(columns={'total_cost': 'scv_cost'})
        
        # Get mixed fleet results
        mixed = self.df[self.df['fleet_type'] == 'MIXED']
        
        # Merge and calculate deltas
        comparison = mixed.merge(scv_baseline, on='instance')
        comparison['cost_delta'] = comparison['total_cost'] - comparison['scv_cost']
        comparison['cost_delta_pct'] = 100.0 * comparison['cost_delta'] / comparison['scv_cost']
        
        # Filter to cases where SCV wins
        scv_wins = comparison[comparison['cost_delta'] > 0].copy()
        scv_wins = scv_wins.sort_values('cost_delta_pct', ascending=False)
        
        return scv_wins
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of SCV wins."""
        total_mixed = len(self.df[self.df['fleet_type'] == 'MIXED'])
        
        return {
            'total_mixed_runs': total_mixed,
            'scv_wins_count': len(self.scv_wins),
            'scv_wins_pct': 100.0 * len(self.scv_wins) / total_mixed,
            'mean_cost_penalty_pct': self.scv_wins['cost_delta_pct'].mean(),
            'max_cost_penalty_pct': self.scv_wins['cost_delta_pct'].max(),
            'median_cost_penalty_pct': self.scv_wins['cost_delta_pct'].median(),
            'total_excess_cost': self.scv_wins['cost_delta'].sum(),
        }
    
    def get_worst_cases(self, n: int = 10) -> pd.DataFrame:
        """Get the n worst cases by cost penalty percentage."""
        return self.scv_wins.nlargest(n, 'cost_delta_pct')[
            ['instance', 'alpha', 'C', 'total_cost', 'scv_cost', 
             'cost_delta', 'cost_delta_pct', 'total_vehicles', 
             'mcv_share', 'total_penalties', 'split_rate']
        ]
    
    def analyze_by_parameter(self) -> Dict:
        """Analyze SCV wins by parameter values."""
        return {
            'by_alpha': self.scv_wins.groupby('alpha').agg({
                'cost_delta_pct': ['count', 'mean', 'max'],
                'cost_delta': 'sum'
            }).round(2),
            'by_C': self.scv_wins.groupby('C').agg({
                'cost_delta_pct': ['count', 'mean', 'max'],
                'cost_delta': 'sum'
            }).round(2),
            'by_instance': self.scv_wins.groupby('instance').agg({
                'cost_delta_pct': ['count', 'mean', 'max'],
                'cost_delta': 'sum'
            }).round(2)
        }
    
    def analyze_patterns(self) -> Dict:
        """Identify patterns in SCV wins."""
        patterns = {}
        
        # Vehicle composition patterns
        patterns['avg_mcv_share'] = self.scv_wins['mcv_share'].mean()
        patterns['avg_total_vehicles'] = self.scv_wins['total_vehicles'].mean()
        patterns['avg_split_rate'] = self.scv_wins['split_rate'].mean()
        patterns['avg_penalties'] = self.scv_wins['total_penalties'].mean()
        
        # Correlation analysis
        correlations = self.scv_wins[
            ['cost_delta_pct', 'alpha', 'C', 'mcv_share', 'split_rate', 'total_penalties']
        ].corr()['cost_delta_pct'].drop('cost_delta_pct')
        patterns['correlations'] = correlations.round(3).to_dict()
        
        return patterns
    
    def get_diagnostic_cases(self, n: int = 5) -> List[Tuple[str, float, float]]:
        """Get specific cases for detailed diagnosis."""
        # Get diverse cases: worst, median, and some typical ones
        cases = []
        
        # Worst case
        worst = self.scv_wins.nlargest(1, 'cost_delta_pct').iloc[0]
        cases.append((worst['instance'], worst['alpha'], worst['C']))
        
        # Median case
        median_idx = len(self.scv_wins) // 2
        median = self.scv_wins.iloc[median_idx]
        cases.append((median['instance'], median['alpha'], median['C']))
        
        # Sample across different penalty ranges
        for pct_range in [(0, 1), (1, 2), (2, 3), (3, 100)]:
            subset = self.scv_wins[
                (self.scv_wins['cost_delta_pct'] > pct_range[0]) & 
                (self.scv_wins['cost_delta_pct'] <= pct_range[1])
            ]
            if not subset.empty:
                sample = subset.sample(n=1).iloc[0]
                cases.append((sample['instance'], sample['alpha'], sample['C']))
        
        return cases[:n]
    
    def select_hypothesis_sample(self, hypothesis: str, n: int = 15) -> List[Tuple[str, float, float]]:
        """
        Select instances most likely affected by a specific hypothesis.
        
        Args:
            hypothesis: One of H1-H6 hypothesis codes
            n: Number of instances to select
            
        Returns:
            List of (instance, alpha, C) tuples
        """
        cases = []
        
        if hypothesis == "H1":  # Solver configuration differences
            # Select cases with high optimality gaps or solver issues
            # First, check if we have optimality gap data
            if 'optimality_gap' in self.scv_wins.columns:
                high_gap = self.scv_wins[self.scv_wins['optimality_gap'] > 0.01]
                if len(high_gap) >= n:
                    sample = high_gap.nlargest(n, 'optimality_gap')
                else:
                    # Supplement with cases that have high cost penalties
                    sample = pd.concat([
                        high_gap,
                        self.scv_wins.nlargest(n - len(high_gap), 'cost_delta_pct')
                    ])
            else:
                # Fallback: use highest cost penalty cases
                sample = self.scv_wins.nlargest(n, 'cost_delta_pct')
                
        elif hypothesis == "H2":  # Vehicle configuration enumeration
            # Select cases with high alpha values
            sample = self.scv_wins[self.scv_wins['alpha'] > 1.7].nlargest(n, 'alpha')
            if len(sample) < n:
                # Supplement with high cost penalty cases
                additional = self.scv_wins[~self.scv_wins.index.isin(sample.index)].nlargest(n - len(sample), 'cost_delta_pct')
                sample = pd.concat([sample, additional])
                
        elif hypothesis == "H3":  # Initial solution quality
            # Select cases with small cost differences (0.5-2%)
            small_diff = self.scv_wins[
                (self.scv_wins['cost_delta_pct'] >= 0.5) & 
                (self.scv_wins['cost_delta_pct'] <= 2.0)
            ]
            if len(small_diff) >= n:
                sample = small_diff.sample(n=n)
            else:
                sample = small_diff
                # Add some from adjacent ranges
                if len(sample) < n:
                    additional = self.scv_wins[
                        (self.scv_wins['cost_delta_pct'] > 2.0) & 
                        (self.scv_wins['cost_delta_pct'] <= 3.0)
                    ].sample(n=min(n - len(sample), len(self.scv_wins) - len(sample)))
                    sample = pd.concat([sample, additional])
                    
        elif hypothesis == "H4":  # Penalty calculation precision
            # Select cases with lowest penalties
            sample = self.scv_wins.nsmallest(n, 'total_penalties')
            
        elif hypothesis == "H5":  # Constraint formulation
            # Select cases with minimal MCV usage
            low_mcv = self.scv_wins[self.scv_wins['mcv_share'] < 0.1]
            if len(low_mcv) >= n:
                sample = low_mcv.nsmallest(n, 'mcv_share')
            else:
                sample = low_mcv
                # Add cases with slightly higher MCV share
                if len(sample) < n:
                    additional = self.scv_wins[
                        (self.scv_wins['mcv_share'] >= 0.1) & 
                        (self.scv_wins['mcv_share'] < 0.2)
                    ].nsmallest(n - len(sample), 'mcv_share')
                    sample = pd.concat([sample, additional])
                    
        elif hypothesis == "H6":  # Cost calculation differences
            # Select worst cases by cost penalty
            sample = self.scv_wins.nlargest(n, 'cost_delta_pct')
            
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
        
        # Convert to list of tuples
        for _, row in sample.iterrows():
            cases.append((row['instance'], row['alpha'], row['C']))
            
        return cases[:n]
    
    def analyze_hypothesis(self, hypothesis: str) -> Dict:
        """
        Analyze characteristics of instances that would be affected by a hypothesis.
        
        Args:
            hypothesis: One of H1-H6 hypothesis codes
            
        Returns:
            Dictionary with analysis results
        """
        sample_cases = self.select_hypothesis_sample(hypothesis, n=100)  # Get larger sample for analysis
        sample_indices = []
        
        # Find indices of sample cases in scv_wins
        for instance, alpha, C in sample_cases:
            mask = (
                (self.scv_wins['instance'] == instance) & 
                (self.scv_wins['alpha'] == alpha) & 
                (self.scv_wins['C'] == C)
            )
            indices = self.scv_wins[mask].index.tolist()
            sample_indices.extend(indices)
        
        if not sample_indices:
            return {"error": "No matching cases found"}
            
        sample_df = self.scv_wins.loc[sample_indices]
        
        analysis = {
            "hypothesis": hypothesis,
            "sample_size": len(sample_df),
            "mean_cost_penalty_pct": sample_df['cost_delta_pct'].mean(),
            "max_cost_penalty_pct": sample_df['cost_delta_pct'].max(),
            "total_excess_cost": sample_df['cost_delta'].sum(),
            "characteristics": {}
        }
        
        # Add hypothesis-specific characteristics
        if hypothesis == "H1":
            if 'optimality_gap' in sample_df.columns:
                analysis["characteristics"]["mean_optimality_gap"] = sample_df['optimality_gap'].mean()
            if 'solver_runtime_sec' in sample_df.columns:
                analysis["characteristics"]["mean_solver_time"] = sample_df['solver_runtime_sec'].mean()
                
        elif hypothesis == "H2":
            analysis["characteristics"]["mean_alpha"] = sample_df['alpha'].mean()
            analysis["characteristics"]["alpha_range"] = (sample_df['alpha'].min(), sample_df['alpha'].max())
            
        elif hypothesis == "H3":
            analysis["characteristics"]["cost_penalty_range"] = (
                sample_df['cost_delta_pct'].min(), 
                sample_df['cost_delta_pct'].max()
            )
            
        elif hypothesis == "H4":
            analysis["characteristics"]["mean_penalties"] = sample_df['total_penalties'].mean()
            analysis["characteristics"]["min_penalties"] = sample_df['total_penalties'].min()
            
        elif hypothesis == "H5":
            analysis["characteristics"]["mean_mcv_share"] = sample_df['mcv_share'].mean()
            analysis["characteristics"]["cases_with_zero_mcv"] = (sample_df['mcv_share'] == 0).sum()
            
        elif hypothesis == "H6":
            analysis["characteristics"]["worst_case_penalty"] = sample_df['cost_delta_pct'].max()
            analysis["characteristics"]["worst_case_instance"] = sample_df.nlargest(1, 'cost_delta_pct').iloc[0]['instance']
        
        return analysis
    
    def detailed_comparison(self, instance: str, alpha: float, C: float) -> Dict:
        """Run detailed comparison between SCV and mixed solutions."""
        # Load demand data
        demand_path = Path(f"data/demand_profiles/{instance}.csv")
        customers_df = load_customer_demand(str(demand_path))
        
        # Run SCV baseline
        scv_params = make_scv_fleet(instance)
        scv_solution = optimize(customers_df, scv_params)
        
        # Run mixed fleet
        mixed_params = make_mixed_fleet(alpha=alpha, C=C, demand_day=instance, allow_split=True)
        mixed_solution = optimize(customers_df, mixed_params)
        
        # Extract detailed comparison
        comparison = {
            'instance': instance,
            'alpha': alpha,
            'C': C,
            'scv_cost': scv_solution.total_cost,
            'mixed_cost': mixed_solution.total_cost,
            'cost_delta': mixed_solution.total_cost - scv_solution.total_cost,
            'cost_delta_pct': 100.0 * (mixed_solution.total_cost - scv_solution.total_cost) / scv_solution.total_cost,
            
            # Vehicle usage
            'scv_vehicles': scv_solution.total_vehicles,
            'mixed_vehicles': mixed_solution.total_vehicles,
            'scv_vehicles_used': scv_solution.vehicles_used,
            'mixed_vehicles_used': mixed_solution.vehicles_used,
            
            # Cost breakdown
            'scv_fixed_cost': scv_solution.total_fixed_cost,
            'mixed_fixed_cost': mixed_solution.total_fixed_cost,
            'scv_variable_cost': scv_solution.total_variable_cost,
            'mixed_variable_cost': mixed_solution.total_variable_cost,
            'scv_penalties': scv_solution.total_penalties,
            'mixed_penalties': mixed_solution.total_penalties,
            
            # Solution quality
            'scv_optimality_gap': scv_solution.optimality_gap,
            'mixed_optimality_gap': mixed_solution.optimality_gap,
            'scv_solver_time': scv_solution.solver_runtime_sec,
            'mixed_solver_time': mixed_solution.solver_runtime_sec,
            'scv_solver_status': scv_solution.solver_status,
            'mixed_solver_status': mixed_solution.solver_status,
            
            # Cluster details
            'scv_clusters': len(scv_solution.selected_clusters) if scv_solution.selected_clusters else 0,
            'mixed_clusters': len(mixed_solution.selected_clusters) if mixed_solution.selected_clusters else 0,
        }
        
        return comparison
    
    def generate_report(self, output_path: Path = None) -> str:
        """Generate comprehensive diagnostic report."""
        if output_path is None:
            output_path = Path("scv_wins_diagnostic_report.md")
        
        report = []
        report.append("# SCV Baseline Wins Diagnostic Report\n")
        
        # Summary statistics
        stats = self.get_summary_stats()
        report.append("## Summary Statistics\n")
        report.append(f"- Total mixed fleet runs: {stats['total_mixed_runs']}")
        report.append(f"- SCV wins: {stats['scv_wins_count']} ({stats['scv_wins_pct']:.1f}%)")
        report.append(f"- Mean cost penalty: {stats['mean_cost_penalty_pct']:.1f}%")
        report.append(f"- Maximum cost penalty: {stats['max_cost_penalty_pct']:.1f}%")
        report.append(f"- Total excess cost: ${stats['total_excess_cost']:.2f}\n")
        
        # Worst cases
        report.append("## Worst Cases\n")
        worst = self.get_worst_cases(10)
        report.append(worst.to_markdown(index=False))
        report.append("")
        
        # Pattern analysis
        patterns = self.analyze_patterns()
        report.append("## Pattern Analysis\n")
        report.append(f"- Average MCV share in problematic cases: {patterns['avg_mcv_share']:.1%}")
        report.append(f"- Average vehicles used: {patterns['avg_total_vehicles']:.1f}")
        report.append(f"- Average split rate: {patterns['avg_split_rate']:.3f}")
        report.append(f"- Average penalties: ${patterns['avg_penalties']:.2f}\n")
        
        report.append("### Correlations with cost penalty:")
        for param, corr in patterns['correlations'].items():
            report.append(f"- {param}: {corr:.3f}")
        report.append("")
        
        # Parameter analysis
        report.append("## Analysis by Parameters\n")
        param_analysis = self.analyze_by_parameter()
        
        report.append("### By Alpha Value")
        report.append(param_analysis['by_alpha'].to_markdown())
        report.append("")
        
        report.append("### By C Value")
        report.append(param_analysis['by_C'].to_markdown())
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        return report_text


def main():
    """Run diagnostic analysis."""
    print("Initializing SCV wins diagnostic...")
    diagnostic = SCVWinsDiagnostic()
    
    # Print summary
    stats = diagnostic.get_summary_stats()
    print(f"\nFound {stats['scv_wins_count']} cases where SCV beats mixed fleet ({stats['scv_wins_pct']:.1f}%)")
    print(f"Mean cost penalty: {stats['mean_cost_penalty_pct']:.1f}%")
    print(f"Max cost penalty: {stats['max_cost_penalty_pct']:.1f}%")
    
    # Generate report
    print("\nGenerating detailed report...")
    report_path = Path("scv_wins_diagnostic_report.md")
    diagnostic.generate_report(report_path)
    print(f"Report saved to: {report_path}")
    
    # Get diagnostic cases for detailed analysis
    print("\nDiagnostic cases for detailed analysis:")
    cases = diagnostic.get_diagnostic_cases()
    for i, (instance, alpha, C) in enumerate(cases, 1):
        print(f"{i}. {instance} with α={alpha:.1f}, C={C:.0f}")
    
    # Optional: Run detailed comparison on worst case
    print("\nRunning detailed comparison on worst case...")
    worst = diagnostic.get_worst_cases(1).iloc[0]
    comparison = diagnostic.detailed_comparison(
        worst['instance'], worst['alpha'], worst['C']
    )
    
    print(f"\nWorst case comparison:")
    print(f"Instance: {comparison['instance']}")
    print(f"Parameters: α={comparison['alpha']}, C={comparison['C']}")
    print(f"SCV cost: ${comparison['scv_cost']:.2f}")
    print(f"Mixed cost: ${comparison['mixed_cost']:.2f}")
    print(f"Cost penalty: {comparison['cost_delta_pct']:.1f}%")
    print(f"Optimality gaps: SCV={comparison['scv_optimality_gap']:.3f}, Mixed={comparison['mixed_optimality_gap']:.3f}")
    print(f"Solver times: SCV={comparison['scv_solver_time']:.1f}s, Mixed={comparison['mixed_solver_time']:.1f}s")


if __name__ == "__main__":
    main()
