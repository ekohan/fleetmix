"""
Generate scatter plots showing MCV savings robustness across demand levels,
with separate visualizations for α and C parameter variations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from experiments.alpha_analysis.full_analysis import load_results

def create_demand_robustness_plots():
    """Create scatter plots showing MCV savings vs demand with parameter variations."""
    
    # Load data
    df_full = load_results()
    
    # Load demand characterization
    char_df = pd.read_csv("results/demand_characterization/daily_summary.csv")
    char_df['day_id'] = 'sales_' + char_df['day_id'] + '_demand'
    char_df = char_df.rename(columns={'total_kg': 'demand_total_kg'})
    
    # Merge
    merged = df_full.merge(char_df[['day_id', 'demand_total_kg']], on='day_id')
    
    # Calculate savings for each configuration
    savings_data = []
    
    for day in merged['day_id'].unique():
        day_data = merged[merged['day_id'] == day]
        scv_cost = day_data[day_data['fleet_type'] == 'SCV']['solver_cost'].values
        
        if len(scv_cost) == 0:
            continue
            
        scv_cost = scv_cost[0]
        demand_kg = day_data['demand_total_kg'].iloc[0]
        
        for (alpha_pct, c_pct), group in day_data[day_data['fleet_type'] == 'MCV'].groupby(['alpha_surcharge_pct', 'c_pct_scv']):
            mcv_cost = group['solver_cost'].values[0]
            savings_pct = (scv_cost - mcv_cost) / scv_cost * 100
            
            savings_data.append({
                'day_id': day,
                'demand_kg': demand_kg,
                'alpha': alpha_pct,
                'C': c_pct,
                'savings_pct': savings_pct
            })
    
    savings_df = pd.DataFrame(savings_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Alpha variations (fix C=10%)
    c_fixed = 10
    alpha_data = savings_df[savings_df['C'] == c_fixed]
    
    # Get unique demand days for x-axis positions
    unique_days = sorted(alpha_data['demand_kg'].unique())
    day_positions = {day: i for i, day in enumerate(unique_days)}
    
    # Prepare data for box plots
    box_data_alpha = []
    positions = []
    
    for demand in unique_days:
        day_savings = alpha_data[alpha_data['demand_kg'] == demand]['savings_pct'].values
        if len(day_savings) > 0:
            box_data_alpha.append(day_savings)
            positions.append(day_positions[demand])
    
    # Create box plots
    bp1 = ax1.boxplot(box_data_alpha, positions=positions, widths=0.6, 
                      patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add scatter points for specific alpha values
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_data['alpha'].unique())))
    alpha_values = sorted(alpha_data['alpha'].unique())
    
    for i, alpha in enumerate(alpha_values):
        alpha_subset = alpha_data[alpha_data['alpha'] == alpha]
        x_positions = [day_positions[d] + np.random.normal(0, 0.1) for d in alpha_subset['demand_kg']]
        ax1.scatter(x_positions, alpha_subset['savings_pct'], 
                   color=colors[i], label=f'α={alpha}%', s=30, alpha=0.6)
    
    ax1.set_xlabel('Total Demand (kg)', fontsize=12)
    ax1.set_ylabel('MCV Savings (%)', fontsize=12)
    ax1.set_title(f'MCV Savings vs Demand: α Variations (C fixed at {c_fixed}%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Set x-axis labels
    ax1.set_xticks(positions[::5])  # Show every 5th label
    ax1.set_xticklabels([f'{int(unique_days[i]/1000)}k' for i in positions[::5]], rotation=45)
    
    # Add trend line for median
    medians = [np.median(data) for data in box_data_alpha]
    ax1.plot(positions, medians, 'r--', alpha=0.5, linewidth=2, label='Median trend')
    
    # Plot 2: C variations (fix α=20%)
    alpha_fixed = 20
    c_data = savings_df[savings_df['alpha'] == alpha_fixed]
    
    # Prepare data for box plots
    box_data_c = []
    positions_c = []
    
    for demand in unique_days:
        day_savings = c_data[c_data['demand_kg'] == demand]['savings_pct'].values
        if len(day_savings) > 0:
            box_data_c.append(day_savings)
            positions_c.append(day_positions[demand])
    
    # Create box plots
    bp2 = ax2.boxplot(box_data_c, positions=positions_c, widths=0.6, 
                      patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    # Add scatter points for specific C values
    colors_c = plt.cm.plasma(np.linspace(0, 1, len(c_data['C'].unique())))
    c_values = sorted(c_data['C'].unique())
    
    for i, c in enumerate(c_values):
        c_subset = c_data[c_data['C'] == c]
        x_positions = [day_positions[d] + np.random.normal(0, 0.1) for d in c_subset['demand_kg']]
        ax2.scatter(x_positions, c_subset['savings_pct'], 
                   color=colors_c[i], label=f'C={c}%', s=30, alpha=0.6)
    
    ax2.set_xlabel('Total Demand (kg)', fontsize=12)
    ax2.set_ylabel('MCV Savings (%)', fontsize=12)
    ax2.set_title(f'MCV Savings vs Demand: C Variations (α fixed at {alpha_fixed}%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Set x-axis labels
    ax2.set_xticks(positions_c[::5])  # Show every 5th label
    ax2.set_xticklabels([f'{int(unique_days[i]/1000)}k' for i in positions_c[::5]], rotation=45)
    
    # Add trend line for median
    medians_c = [np.median(data) for data in box_data_c]
    ax2.plot(positions_c, medians_c, 'r--', alpha=0.5, linewidth=2, label='Median trend')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_heatmaps/demand_robustness_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and print summary statistics
    print("\n" + "="*60)
    print("DEMAND ROBUSTNESS SUMMARY")
    print("="*60)
    
    # For alpha variations
    print(f"\nAlpha variations (C={c_fixed}%):")
    for alpha in sorted(alpha_data['alpha'].unique()):
        subset = alpha_data[alpha_data['alpha'] == alpha]
        print(f"  α={alpha}%: mean={subset['savings_pct'].mean():.1f}%, std={subset['savings_pct'].std():.1f}%")
    
    # For C variations  
    print(f"\nC variations (α={alpha_fixed}%):")
    for c in sorted(c_data['C'].unique()):
        subset = c_data[c_data['C'] == c]
        print(f"  C={c}%: mean={subset['savings_pct'].mean():.1f}%, std={subset['savings_pct'].std():.1f}%")
    
    # Overall correlation with demand
    print("\nCorrelation with demand:")
    print(f"  Overall: r={savings_df[['demand_kg', 'savings_pct']].corr().iloc[0,1]:.3f}")
    print(f"  At α=0%, C=0%: r={savings_df[(savings_df['alpha']==0) & (savings_df['C']==0)][['demand_kg', 'savings_pct']].corr().iloc[0,1]:.3f}")
    print(f"  At α={alpha_fixed}%, C={c_fixed}%: r={savings_df[(savings_df['alpha']==alpha_fixed) & (savings_df['C']==c_fixed)][['demand_kg', 'savings_pct']].corr().iloc[0,1]:.3f}")
    
    # Create a simplified version with just min/max ranges
    create_simplified_range_plot(savings_df)

def create_simplified_range_plot(savings_df):
    """Create a simplified version showing just the range of savings."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Range across all alpha values (C=10%)
    c_fixed = 10
    alpha_range = savings_df[savings_df['C'] == c_fixed]
    
    # Group by demand and get min/max
    demand_summary = []
    for demand, group in alpha_range.groupby('demand_kg'):
        demand_summary.append({
            'demand_kg': demand,
            'min_savings': group['savings_pct'].min(),
            'max_savings': group['savings_pct'].max(),
            'mean_savings': group['savings_pct'].mean(),
            'range': group['savings_pct'].max() - group['savings_pct'].min()
        })
    
    summary_df = pd.DataFrame(demand_summary).sort_values('demand_kg')
    
    # Plot range
    ax1.fill_between(summary_df['demand_kg'], 
                     summary_df['min_savings'], 
                     summary_df['max_savings'],
                     alpha=0.3, color='blue', label='Range across α values')
    ax1.plot(summary_df['demand_kg'], summary_df['mean_savings'], 
             'b-', linewidth=2, label='Mean')
    
    ax1.set_xlabel('Total Demand (kg)')
    ax1.set_ylabel('MCV Savings (%)')
    ax1.set_title(f'MCV Savings Range: α ∈ [0%, 100%], C = {c_fixed}%')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add text showing average range
    avg_range = summary_df['range'].mean()
    ax1.text(0.02, 0.98, f'Average range: {avg_range:.1f} pp', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Range across all C values (α=20%)
    alpha_fixed = 20
    c_range = savings_df[savings_df['alpha'] == alpha_fixed]
    
    # Group by demand and get min/max
    demand_summary_c = []
    for demand, group in c_range.groupby('demand_kg'):
        demand_summary_c.append({
            'demand_kg': demand,
            'min_savings': group['savings_pct'].min(),
            'max_savings': group['savings_pct'].max(),
            'mean_savings': group['savings_pct'].mean(),
            'range': group['savings_pct'].max() - group['savings_pct'].min()
        })
    
    summary_df_c = pd.DataFrame(demand_summary_c).sort_values('demand_kg')
    
    # Plot range
    ax2.fill_between(summary_df_c['demand_kg'], 
                     summary_df_c['min_savings'], 
                     summary_df_c['max_savings'],
                     alpha=0.3, color='green', label='Range across C values')
    ax2.plot(summary_df_c['demand_kg'], summary_df_c['mean_savings'], 
             'g-', linewidth=2, label='Mean')
    
    ax2.set_xlabel('Total Demand (kg)')
    ax2.set_ylabel('MCV Savings (%)')
    ax2.set_title(f'MCV Savings Range: α = {alpha_fixed}%, C ∈ [0%, 50%]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add text showing average range
    avg_range_c = summary_df_c['range'].mean()
    ax2.text(0.02, 0.98, f'Average range: {avg_range_c:.1f} pp', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/enhanced_heatmaps/demand_robustness_ranges.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("RANGE ANALYSIS")
    print("="*60)
    print(f"α variations (C={c_fixed}%): average range = {avg_range:.1f} pp")
    print(f"C variations (α={alpha_fixed}%): average range = {avg_range_c:.1f} pp")
    print(f"Ratio of ranges: {avg_range/avg_range_c:.2f}")

def main():
    create_demand_robustness_plots()
    print("\n" + "="*60)
    print("Plots saved to results/enhanced_heatmaps/")
    print("- demand_robustness_scatter.png (detailed with box plots)")
    print("- demand_robustness_ranges.png (simplified range view)")
    print("="*60)

if __name__ == "__main__":
    main() 