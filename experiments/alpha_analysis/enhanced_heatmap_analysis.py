"""
Enhanced heatmap analysis with RSM overlays, win-probability modeling,
demand stratification, and safe zones visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats
from experiments.alpha_analysis.full_analysis import load_results, compute_stats, get_savings_df
from experiments.alpha_analysis.hte_analysis import load_and_prepare_data, _center_scale

# Output directory
OUTPUT_DIR = Path("results/enhanced_heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fit_rsm_for_overlay(df):
    """Fit the RSM model for overlay purposes."""
    # Using the standard 5-term RSM
    model = smf.ols('cost_diff ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C', data=df).fit()
    return model

def fit_win_probability_model(df):
    """Fit logistic regression for P(win)."""
    # Create binary win variable
    df['win'] = (df['cost_diff'] > 0).astype(int)
    
    # Fit logit model
    logit_model = smf.logit('win ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C', data=df).fit()
    print("\nWin Probability Model Summary:")
    print(logit_model.summary())
    
    return logit_model

def enhancement_1_rsm_overlay(df, stats, output_dir):
    """Enhancement 1: Overlay the modelled break-even frontier on the heatmap."""
    print("\n=== Enhancement 1: RSM Break-even Overlay ===")
    
    # Fit RSM model
    rsm_model = fit_rsm_for_overlay(df)
    
    # Create grid for predictions
    alpha_range = np.linspace(1.0, 2.0, 200)
    c_range = np.linspace(0, 50, 200)
    A_grid, C_grid = np.meshgrid(alpha_range, c_range)
    
    # Predict on grid
    pred_df = pd.DataFrame({
        'alpha': A_grid.ravel(),
        'C': C_grid.ravel()
    })
    Z_pred = rsm_model.predict(pred_df).values.reshape(A_grid.shape)
    
    # Convert to percentage basis for consistency
    alpha_pct_grid = (A_grid - 1) * 100
    c_pct_grid = C_grid  # Already in percentage
    
    # Create the base heatmap
    pivot_pct = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').round(1)
    pivot_wr = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='win_rate')
    
    days = int(stats['num_days'].max())
    annot = pd.DataFrame(index=pivot_pct.index, columns=pivot_pct.columns, dtype=object)
    for a in pivot_pct.index:
        for c in pivot_pct.columns:
            val = pivot_pct.loc[a, c]
            if pd.isna(val):
                annot.loc[a, c] = ""
            else:
                wins = int(round(pivot_wr.loc[a, c] * days))
                annot.loc[a, c] = f"{val:.0f}%\n{wins}/{days}"
    
    # Plot with overlay
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_pct,
        annot=annot,
        fmt="",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Average %-Savings vs SCV'},
        ax=ax
    )
    
    # Overlay break-even contour (where Z_pred = 0)
    contour = ax.contour(c_pct_grid, alpha_pct_grid, Z_pred, 
                        levels=[0], colors='black', linewidths=2.5, linestyles='dashed')
    ax.clabel(contour, inline=True, fontsize=10, fmt='Break-even')
    
    # Add ±5% contours
    contour_5 = ax.contour(c_pct_grid, alpha_pct_grid, Z_pred, 
                          levels=[-5, 5], colors='gray', linewidths=1.5, linestyles='dotted', alpha=0.7)
    ax.clabel(contour_5, inline=True, fontsize=8, fmt='%+d%%')
    
    ax.set_title("MCV Advantage with RSM Break-even Frontier Overlay")
    ax.set_ylabel("MCV Surcharge % (from α)")
    ax.set_xlabel("C as % of SCV fixed cost")
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_1_rsm_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

def enhancement_3_win_probability(df, stats, output_dir):
    """Enhancement 3: Layer in the win-probability model."""
    print("\n=== Enhancement 3: Win Probability Model ===")
    
    # Fit logit model
    logit_model = fit_win_probability_model(df)
    
    # Create grid for predictions
    alpha_range = np.linspace(1.0, 2.0, 200)
    c_range = np.linspace(0, 50, 200)
    A_grid, C_grid = np.meshgrid(alpha_range, c_range)
    
    pred_df = pd.DataFrame({
        'alpha': A_grid.ravel(),
        'C': C_grid.ravel()
    })
    
    # Predict probabilities
    prob_pred = logit_model.predict(pred_df).values.reshape(A_grid.shape)
    
    # Convert to percentage basis
    alpha_pct_grid = (A_grid - 1) * 100
    c_pct_grid = C_grid
    
    # Create enhanced heatmap with probability contours
    pivot_pct = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').round(1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Average %-Savings vs SCV'},
        ax=ax,
        annot_kws={'size': 9}
    )
    
    # Overlay probability contours
    prob_contours = ax.contour(c_pct_grid, alpha_pct_grid, prob_pred,
                               levels=[0.5, 0.75, 0.9, 0.95],
                               colors='blue', linewidths=[1, 1.5, 2, 2.5],
                               linestyles=[':', '--', '-', '-'])
    ax.clabel(prob_contours, inline=True, fontsize=9, 
              fmt=lambda x: f'P={x:.0%}')
    
    ax.set_title("MCV Advantage with Win Probability Contours")
    ax.set_ylabel("MCV Surcharge % (from α)")
    ax.set_xlabel("C as % of SCV fixed cost")
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_3_win_probability.png', dpi=300, bbox_inches='tight')
    plt.close()

def enhancement_4a_demand_multiples(df, output_dir):
    """Enhancement 4a: Small multiples by demand strata."""
    print("\n=== Enhancement 4a: Demand-Stratified Small Multiples ===")
    
    # Load full results dataset
    df_full = load_results()
    
    # Load demand characterization
    char_df = pd.read_csv("results/demand_characterization/daily_summary.csv")
    char_df['day_id'] = 'sales_' + char_df['day_id'] + '_demand'
    char_df = char_df.rename(columns={'num_customers': 'demand_customers', 'total_kg': 'demand_total_kg'})
    
    # Merge with full data
    merged = df_full.merge(char_df[['day_id', 'demand_total_kg']], on='day_id')
    
    # Create demand terciles
    merged['demand_tercile'] = pd.qcut(merged['demand_total_kg'], 3, labels=['Low', 'Medium', 'High'])
    
    # Debug prints
    print(f"Total merged records: {len(merged)}")
    print(f"Demand tercile distribution: {merged['demand_tercile'].value_counts().to_dict()}")
    print(f"Fleet types in merged: {merged['fleet_type'].value_counts().to_dict()}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))
    
    for i, (demand_level, ax) in enumerate(zip(['Low', 'Medium', 'High'], axes)):
        # Filter data for this demand level - but ensure we have corresponding days
        level_days = merged[merged['demand_tercile'] == demand_level]['day_id'].unique()
        level_data = df_full[df_full['day_id'].isin(level_days)]
        
        print(f"\n{demand_level} demand: {len(level_days)} days, {len(level_data)} records")
        print(f"Fleet types: {level_data['fleet_type'].value_counts().to_dict()}")
        
        # Compute stats for this subset
        level_stats = compute_stats(level_data)
        
        if len(level_stats) == 0:
            print(f"Warning: No stats computed for {demand_level} demand level")
            ax.text(0.5, 0.5, f'No data for {demand_level} demand level', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Create heatmap
        pivot_pct = level_stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').round(1)
        pivot_wr = level_stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='win_rate')
        
        days_level = int(level_stats['num_days'].max())
        annot = pd.DataFrame(index=pivot_pct.index, columns=pivot_pct.columns, dtype=object)
        for a in pivot_pct.index:
            for c in pivot_pct.columns:
                val = pivot_pct.loc[a, c]
                if pd.isna(val):
                    annot.loc[a, c] = ""
                else:
                    wins = int(round(pivot_wr.loc[a, c] * days_level))
                    annot.loc[a, c] = f"{val:.0f}%\n{wins}/{days_level}"
        
        sns.heatmap(
            pivot_pct,
            annot=annot,
            fmt="",
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Avg %-Savings'},
            ax=ax,
            vmin=-30, vmax=30  # Keep scale consistent
        )
        
        # Calculate average demand for this level
        level_demand_info = merged[merged['demand_tercile'] == demand_level]
        avg_demand = level_demand_info['demand_total_kg'].mean()
        ax.set_title(f'{demand_level} Demand Days (avg: {avg_demand:.0f} kg)')
        ax.set_ylabel("MCV Surcharge %")
        if i == 2:  # Only label x-axis on bottom plot
            ax.set_xlabel("C as % of SCV fixed cost")
        else:
            ax.set_xlabel("")
    
    plt.suptitle("MCV Advantage Across Demand Levels", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_4a_demand_multiples.png', dpi=300, bbox_inches='tight')
    plt.close()

def enhancement_7_safe_zones(df, stats, output_dir):
    """Enhancement 7: Convey managerial 'safe zones'."""
    print("\n=== Enhancement 7: Safe Zones Visualization ===")
    
    # Define safe zone criteria
    WIN_PROB_THRESHOLD = 0.9
    MIN_SAVINGS_THRESHOLD = 5.0
    
    # Get detailed data for each cell from the full dataset
    df_full = load_results()
    savings_df = get_savings_df(df_full)
    
    # Calculate metrics for each (alpha, C) combination
    safe_zone_data = []
    for (alpha_pct, c_pct), group in savings_df.groupby(['alpha_surcharge_pct', 'c_pct_scv']):
        win_prob = (group['pct_savings'] > 0).mean()
        avg_savings = group['pct_savings'].mean()
        
        is_safe = (win_prob >= WIN_PROB_THRESHOLD) and (avg_savings >= MIN_SAVINGS_THRESHOLD)
        
        safe_zone_data.append({
            'alpha_surcharge_pct': alpha_pct,
            'c_pct_scv': c_pct,
            'win_prob': win_prob,
            'avg_savings': avg_savings,
            'is_safe': is_safe
        })
    
    safe_df = pd.DataFrame(safe_zone_data)
    
    # Create the visualization
    pivot_pct = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').round(1)
    pivot_wr = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='win_rate')
    
    days = int(stats['num_days'].max())
    annot = pd.DataFrame(index=pivot_pct.index, columns=pivot_pct.columns, dtype=object)
    for a in pivot_pct.index:
        for c in pivot_pct.columns:
            val = pivot_pct.loc[a, c]
            if pd.isna(val):
                annot.loc[a, c] = ""
            else:
                wins = int(round(pivot_wr.loc[a, c] * days))
                # Check if this cell is in safe zone
                is_safe = safe_df[(safe_df['alpha_surcharge_pct'] == a) & 
                                 (safe_df['c_pct_scv'] == c)]['is_safe'].values
                if len(is_safe) > 0 and is_safe[0]:
                    annot.loc[a, c] = f"{val:.0f}%\n{wins}/{days}\n✓"
                else:
                    annot.loc[a, c] = f"{val:.0f}%\n{wins}/{days}"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_pct,
        annot=annot,
        fmt="",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Average %-Savings vs SCV'},
        ax=ax
    )
    
    # Add safe zone shading
    for _, row in safe_df[safe_df['is_safe']].iterrows():
        # Find the position in the heatmap
        try:
            y_idx = list(pivot_pct.index).index(row['alpha_surcharge_pct'])
            x_idx = list(pivot_pct.columns).index(row['c_pct_scv'])
            
            # Add a rectangle with light green overlay
            rect = patches.Rectangle((x_idx, y_idx), 1, 1, 
                                   linewidth=3, edgecolor='darkgreen',
                                   facecolor='none', linestyle='-')
            ax.add_patch(rect)
        except ValueError:
            continue
    
    # Add legend for safe zone
    safe_patch = patches.Patch(color='darkgreen', alpha=0.3, 
                              label=f'Safe Zone: P(win)≥{WIN_PROB_THRESHOLD:.0%} & Avg Savings≥{MIN_SAVINGS_THRESHOLD}%')
    ax.legend(handles=[safe_patch], loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_title("MCV Advantage with Managerial Safe Zones")
    ax.set_ylabel("MCV Surcharge % (from α)")
    ax.set_xlabel("C as % of SCV fixed cost")
    
    # Add text box with safe zone explanation
    textstr = f"✓ = Safe Zone\n(Win rate ≥ {WIN_PROB_THRESHOLD:.0%} AND\nAvg savings ≥ {MIN_SAVINGS_THRESHOLD}%)"
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_7_safe_zones.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("ENHANCED HEATMAP ANALYSIS")
    print("="*60)
    
    # Load base results
    print("\nLoading results...")
    df_full = load_results()
    
    # Prepare data as in HTE analysis
    df = load_and_prepare_data()
    
    # Compute stats for heatmaps
    stats = compute_stats(df_full)
    
    # Run each enhancement
    enhancement_1_rsm_overlay(df, stats, OUTPUT_DIR)
    enhancement_3_win_probability(df, stats, OUTPUT_DIR)
    enhancement_4a_demand_multiples(df, OUTPUT_DIR)
    enhancement_7_safe_zones(df, stats, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ENHANCED ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main() 