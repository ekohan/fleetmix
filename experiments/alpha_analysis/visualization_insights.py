"""
Extract and summarize key insights from enhanced heatmap visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from experiments.alpha_analysis.full_analysis import load_results, compute_stats
from experiments.alpha_analysis.hte_analysis import load_and_prepare_data

def extract_key_insights():
    """Extract key quantitative insights from the analysis."""
    
    # Load data
    df_full = load_results()
    df_hte = load_and_prepare_data()
    stats = compute_stats(df_full)
    
    insights = {}
    
    # 1. Break-even frontier characteristics
    # Find approximate break-even points
    break_even_points = []
    for c in sorted(stats['c_pct_scv'].unique()):
        c_data = stats[stats['c_pct_scv'] == c]
        # Find where average savings crosses zero
        positive = c_data[c_data['avg_pct_savings'] > 0]
        if len(positive) > 0:
            max_alpha = positive['alpha_surcharge_pct'].max()
            break_even_points.append({'C': c, 'max_alpha': max_alpha})
    
    insights['break_even_frontier'] = pd.DataFrame(break_even_points)
    
    # 2. Safe zone analysis
    safe_zones = []
    for (alpha, c), group in stats.groupby(['alpha_surcharge_pct', 'c_pct_scv']):
        if group['win_rate'].iloc[0] >= 0.9 and group['avg_pct_savings'].iloc[0] >= 5:
            safe_zones.append({
                'alpha': alpha,
                'C': c,
                'win_rate': group['win_rate'].iloc[0],
                'avg_savings': group['avg_pct_savings'].iloc[0]
            })
    
    insights['safe_zones'] = pd.DataFrame(safe_zones)
    
    # 3. Demand heterogeneity impact
    # Load demand characterization
    char_df = pd.read_csv("results/demand_characterization/daily_summary.csv")
    char_df['day_id'] = 'sales_' + char_df['day_id'] + '_demand'
    char_df = char_df.rename(columns={'total_kg': 'demand_total_kg'})
    
    merged = df_full.merge(char_df[['day_id', 'demand_total_kg']], on='day_id')
    merged['demand_tercile'] = pd.qcut(merged['demand_total_kg'], 3, labels=['Low', 'Medium', 'High'])
    
    demand_impact = []
    for level in ['Low', 'Medium', 'High']:
        level_days = merged[merged['demand_tercile'] == level]['day_id'].unique()
        level_data = df_full[df_full['day_id'].isin(level_days)]
        level_stats = compute_stats(level_data)
        
        # Find max viable alpha at C=10
        c10_data = level_stats[level_stats['c_pct_scv'] == 10]
        viable = c10_data[c10_data['avg_pct_savings'] > 0]
        max_alpha = viable['alpha_surcharge_pct'].max() if len(viable) > 0 else 0
        
        demand_impact.append({
            'demand_level': level,
            'avg_demand_kg': merged[merged['demand_tercile'] == level]['demand_total_kg'].mean(),
            'max_viable_alpha_at_C10': max_alpha
        })
    
    insights['demand_impact'] = pd.DataFrame(demand_impact)
    
    # 4. Key statistics at flagship setting (α=20%, C=10%)
    flagship = stats[(stats['alpha_surcharge_pct'] == 20) & (stats['c_pct_scv'] == 10)]
    if len(flagship) > 0:
        insights['flagship_stats'] = {
            'win_rate': flagship['win_rate'].iloc[0],
            'avg_savings': flagship['avg_pct_savings'].iloc[0],
            'num_days': flagship['num_days'].iloc[0]
        }
    
    # 5. Maximum observed benefits
    max_benefit = stats.loc[stats['avg_pct_savings'].idxmax()]
    insights['max_benefit'] = {
        'alpha': max_benefit['alpha_surcharge_pct'],
        'C': max_benefit['c_pct_scv'],
        'avg_savings': max_benefit['avg_pct_savings'],
        'win_rate': max_benefit['win_rate']
    }
    
    return insights

def print_insights_summary(insights):
    """Print a formatted summary of key insights."""
    
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM ENHANCED VISUALIZATIONS")
    print("="*60)
    
    print("\n1. BREAK-EVEN FRONTIER")
    print("-" * 40)
    if 'break_even_frontier' in insights:
        for _, row in insights['break_even_frontier'].iterrows():
            print(f"At C={row['C']}%: MCV viable up to α={row['max_alpha']}% surcharge")
    
    print("\n2. MANAGERIAL SAFE ZONES (≥90% win rate & ≥5% savings)")
    print("-" * 40)
    if 'safe_zones' in insights and len(insights['safe_zones']) > 0:
        print(f"Found {len(insights['safe_zones'])} safe configurations:")
        print(f"Max safe α: {insights['safe_zones']['alpha'].max()}%")
        print(f"Max safe C: {insights['safe_zones']['C'].max()}%")
    else:
        print("No safe zones found with current criteria")
    
    print("\n3. DEMAND HETEROGENEITY IMPACT")
    print("-" * 40)
    if 'demand_impact' in insights:
        for _, row in insights['demand_impact'].iterrows():
            print(f"{row['demand_level']} demand (avg {row['avg_demand_kg']:.0f} kg): "
                  f"viable up to α={row['max_viable_alpha_at_C10']}% at C=10%")
    
    print("\n4. FLAGSHIP CONFIGURATION (α=20%, C=10%)")
    print("-" * 40)
    if 'flagship_stats' in insights:
        stats = insights['flagship_stats']
        print(f"Win rate: {stats['win_rate']:.1%}")
        print(f"Average savings: {stats['avg_savings']:.1f}%")
        print(f"Evaluated on: {stats['num_days']} days")
    
    print("\n5. MAXIMUM OBSERVED BENEFIT")
    print("-" * 40)
    if 'max_benefit' in insights:
        mb = insights['max_benefit']
        print(f"Configuration: α={mb['alpha']}%, C={mb['C']}%")
        print(f"Average savings: {mb['avg_savings']:.1f}%")
        print(f"Win rate: {mb['win_rate']:.1%}")
    
    print("\n" + "="*60)

def main():
    insights = extract_key_insights()
    print_insights_summary(insights)
    
    # Save insights to CSV
    output_dir = Path("results/enhanced_heatmaps")
    for key, value in insights.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"insights_{key}.csv", index=False)
        elif isinstance(value, dict):
            pd.DataFrame([value]).to_csv(output_dir / f"insights_{key}.csv", index=False)

if __name__ == "__main__":
    main() 