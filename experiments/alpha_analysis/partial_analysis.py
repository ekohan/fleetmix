"""
Analyze partial results from the alpha analysis grid search.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import dataclasses
from fleetmix.config import load_fleetmix_params
import math

BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config.yaml")

RESULTS_RAW = Path("results_raw_v4")
FIGS_DIR = Path("results/partial_figs_v4")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

def _bp(value: float) -> str:
    """Format a ratio (1 = 100%) as basis points string."""
    return f"{value * 10000:.0f} bp"

def load_partial_results():
    """Load all existing JSON results."""
    all_results = []
    json_files = list(RESULTS_RAW.glob("sales_*_demand_*.json"))
    
    print(f"Found {len(json_files)} result files")
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except json.JSONDecodeError:
            print(f"Skipping corrupted file: {json_path}")
            continue
    
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} results")
    if len(df) > 0:
        print(f"Fleet types: {df['fleet_type'].value_counts().to_dict()}")
        print(
            f"Alpha range: {df['alpha'].min():.2f} – {df['alpha'].max():.2f}  | "
            f"Surcharge: {df['alpha_surcharge_pct'].min():.0f}% – {df['alpha_surcharge_pct'].max():.0f}%"
        )
        print(
            f"C range: {df['C'].min():.0f} – {df['C'].max():.0f}  | "
            f"C as % of SCV: {df['c_pct_scv'].min():.0f}% – {df['c_pct_scv'].max():.0f}%"
        )
        print(f"Unique days: {df['day_id'].nunique()}")
    
    # Load base parameters once to obtain SCV fixed cost
    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    base_vehicle_spec = next(iter(base_params.problem.vehicles.values()))
    f_sc = float(base_vehicle_spec.fixed_cost)
    
    print(f"Using SCV fixed cost f_SC = {f_sc}")
    
    # Compute unit-free expressions (vectorised, no branching)
    df['alpha_surcharge_pct'] = ((df['alpha'] - 1) * 100).round().astype(int)
    df['c_pct_scv'] = (100 * df['C'] / f_sc).round().astype(int)
    
    # Filter SCV data for baseline calculations
    scv_df = df[df['fleet_type'] == 'SCV']
    
    if len(scv_df) > 0:
        scv_mean_cost = scv_df['solver_cost'].mean()
        scv_mean_kg = scv_df['cost_per_kg'].mean()
        scv_mean_drop = scv_df['cost_per_drop'].mean()
        
        df['cost_index'] = df['solver_cost'] / scv_mean_cost
        df['cost_per_kg_index'] = df['cost_per_kg'] / scv_mean_kg
        df['cost_per_drop_index'] = df['cost_per_drop'] / scv_mean_drop
    
    return df

def plot_cost_trends(df):
    """Plot cost trends vs alpha."""
    plt.figure(figsize=(15, 10))
    
    mcv_df = df[df['fleet_type'] == 'MCV']
    scv_df = df[df['fleet_type'] == 'SCV']
    
    # Subplot 1: Raw costs
    plt.subplot(2, 3, 1)
    if len(scv_df) > 0:
        scv_mean = scv_df['solver_cost'].mean()
        plt.axhline(scv_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'SCV mean: {scv_mean:.0f}')
    
    # Show different C values with different colors
    c_values = sorted(mcv_df['C'].unique())[:8]  # Show up to 8 C values
    colors = plt.cm.viridis(np.linspace(0, 1, len(c_values)))
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['solver_cost'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Total Solver Cost')
    plt.title('Total Cost vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cost per kg
    plt.subplot(2, 3, 2)
    if len(scv_df) > 0:
        scv_mean_kg = scv_df['cost_per_kg'].mean()
        plt.axhline(scv_mean_kg, color='red', linestyle='--', linewidth=2,
                   label=f'SCV mean: {scv_mean_kg:.3f}')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['cost_per_kg'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Cost per kg')
    plt.title('Cost per kg vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cost per drop
    plt.subplot(2, 3, 3)
    if len(scv_df) > 0:
        scv_mean_drop = scv_df['cost_per_drop'].mean()
        plt.axhline(scv_mean_drop, color='red', linestyle='--', linewidth=2,
                   label=f'SCV mean: {scv_mean_drop:.2f}')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['cost_per_drop'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Cost per drop')
    plt.title('Cost per drop vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Average visits per customer (replacing split rate)
    plt.subplot(2, 3, 4)
    # Add SCV horizontal line for comparison
    if len(scv_df) > 0:
        scv_mean_visits = scv_df['avg_visits_per_customer'].mean()
        plt.axhline(scv_mean_visits, color='red', linestyle='--', linewidth=2,
                   label=f'SCV mean: {scv_mean_visits:.2f}')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['avg_visits_per_customer'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Avg Visits per Customer')
    plt.title('Avg Visits per Customer vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: MCV advantage (cost difference from SCV)
    plt.subplot(2, 3, 5)
    if len(scv_df) > 0:
        plt.axhline(0, color='red', linestyle='--', linewidth=2,
                   label='Break-even (MCV = SCV)')
        
        for c_val, color in zip(c_values, colors):
            c_data = mcv_df[mcv_df['C'] == c_val]
            if len(c_data) > 1:
                # Calculate advantage: negative means MCV is better
                advantage = []
                surcharges = []
                for _, row in c_data.iterrows():
                    day_scv = scv_df[scv_df['day_id'] == row['day_id']]
                    if len(day_scv) > 0:
                        scv_cost = day_scv['solver_cost'].iloc[0]
                        advantage.append(row['solver_cost'] - scv_cost)
                        surcharges.append(row['alpha_surcharge_pct'])
                
                if len(advantage) > 0:
                    plt.scatter(surcharges, advantage, alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', 
                               color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('MCV - SCV Cost (negative = MCV better)')
    plt.title('MCV Advantage vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Number of customers vs demand
    plt.subplot(2, 3, 6)
    plt.scatter(df['num_customers'], df['total_demand_kg'], 
               c=df['solver_cost'], alpha=0.7, cmap='viridis')
    plt.colorbar(label='Solver Cost')
    plt.xlabel('Number of Customers')
    plt.ylabel('Total Demand (kg)')
    plt.title('Problem Size vs Cost')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'partial_cost_trends_reexpressed.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_index_trends(df):
    """Plot cost index trends vs surcharge."""
    plt.figure(figsize=(15, 10))
    
    mcv_df = df[df['fleet_type'] == 'MCV']
    scv_df = df[df['fleet_type'] == 'SCV']
    
    # Subplot 1: Cost index
    plt.subplot(2, 3, 1)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, 
                label='SCV baseline')
    
    c_values = sorted(mcv_df['C'].unique())[:8]
    colors = plt.cm.viridis(np.linspace(0, 1, len(c_values)))
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['cost_index'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Total Cost Index (rel. SCV)')
    plt.title('Total Cost Index vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cost per kg index
    plt.subplot(2, 3, 2)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, 
                label='SCV baseline')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['cost_per_kg_index'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Cost per kg Index (rel. SCV)')
    plt.title('Cost per kg Index vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cost per drop index
    plt.subplot(2, 3, 3)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, 
                label='SCV baseline')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['cost_per_drop_index'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Cost per drop Index (rel. SCV)')
    plt.title('Cost per drop Index vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Avg visits (same as in cost trends, but using surcharge)
    plt.subplot(2, 3, 4)
    if len(scv_df) > 0:
        scv_mean_visits = scv_df['avg_visits_per_customer'].mean()
        plt.axhline(scv_mean_visits, color='red', linestyle='--', linewidth=2,
                    label=f'SCV mean: {scv_mean_visits:.2f}')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            pct = c_data['c_pct_scv'].iloc[0]
            plt.scatter(c_data['alpha_surcharge_pct'], c_data['avg_visits_per_customer'], 
                       alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Avg Visits per Customer')
    plt.title('Avg Visits per Customer vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Savings % (negative advantage as %)
    plt.subplot(2, 3, 5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2,
                label='Break-even')
    
    for c_val, color in zip(c_values, colors):
        c_data = mcv_df[mcv_df['C'] == c_val]
        if len(c_data) > 1:
            savings_pct = []
            surcharges = []
            for _, row in c_data.iterrows():
                day_scv = scv_df[scv_df['day_id'] == row['day_id']]
                if len(day_scv) > 0:
                    scv_cost = day_scv['solver_cost'].iloc[0]
                    savings = (scv_cost - row['solver_cost']) / scv_cost * 100
                    savings_pct.append(savings)
                    surcharges.append(row['alpha_surcharge_pct'])
            
            if len(savings_pct) > 0:
                plt.scatter(surcharges, savings_pct, alpha=0.7, label=f'C={c_val:.0f} ({pct:.0f}% SCV)', 
                            color=color, s=20)
    
    plt.xlabel('MCV Surcharge (%)  (derived from α)')
    plt.ylabel('Savings % (positive = MCV better)')
    plt.title('MCV Savings vs Surcharge')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Optional, e.g., coverage or something else
    plt.subplot(2, 3, 6)
    # For now, leave empty or add problem size as in cost trends
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'partial_index_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_fleet_comparison(df):
    """Compare MCV vs SCV performance."""
    plt.figure(figsize=(15, 5))
    
    metrics = ['solver_cost', 'cost_per_kg', 'cost_per_drop']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        # Create separate boxplots for each fleet type
        data_to_plot = []
        labels = []
        
        if len(df[df['fleet_type'] == 'SCV']) > 0:
            data_to_plot.append(df[df['fleet_type'] == 'SCV'][metric])
            labels.append('SCV')
        
        if len(df[df['fleet_type'] == 'MCV']) > 0:
            data_to_plot.append(df[df['fleet_type'] == 'MCV'][metric])
            labels.append('MCV')
        
        if data_to_plot:
            plt.boxplot(data_to_plot, tick_labels=labels)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'partial_fleet_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def coverage_analysis(df):
    """Heat-map of observed (α surcharge, C %) coverage."""
    mcv_df = df[df['fleet_type'] == 'MCV']

    if len(mcv_df) == 0:
        print("No MCV data found for coverage analysis")
        return

    rows_label = 'alpha_surcharge_pct'
    cols_label = 'c_pct_scv'
    alpha_vals = sorted(mcv_df[rows_label].unique())
    c_vals = sorted(mcv_df[cols_label].unique())
    coverage = mcv_df.groupby([rows_label, cols_label]).size().unstack(fill_value=0)
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        coverage.reindex(index=alpha_vals, columns=c_vals),
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of days'}
    )
    plt.title('Data Coverage: days per (Surcharge %, C % of SCV)')
    plt.xlabel('C as % of SCV fixed cost')
    plt.ylabel('MCV surcharge % (from α)')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'partial_coverage_pct.png', dpi=150, bbox_inches='tight')
    plt.close()

def summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nOverall dataset:")
    print(f"  Total results: {len(df)}")
    print(f"  Fleet types: {df['fleet_type'].value_counts().to_dict()}")
    print(f"  Unique days: {df['day_id'].nunique()}")
    
    if len(df[df['fleet_type'] == 'MCV']) > 0:
        mcv_df = df[df['fleet_type'] == 'MCV']
        print(f"\nMCV parameter ranges:")
        print(f"  Alpha: {mcv_df['alpha'].min():.2f} - {mcv_df['alpha'].max():.2f}")
        print(f"  C: {mcv_df['C'].min():.0f} - {mcv_df['C'].max():.0f}")
        print(f"  Unique (alpha, C) combinations: {mcv_df[['alpha', 'C']].drop_duplicates().shape[0]}")
    
    print(f"\nCost statistics by fleet type:")
    cost_stats = df.groupby('fleet_type')[['solver_cost', 'cost_per_kg', 'cost_per_drop']].agg(['mean', 'std', 'min', 'max'])
    print(cost_stats)
    idx_stats = df.groupby('fleet_type')[['cost_index', 'cost_per_kg_index', 'cost_per_drop_index']].mean()
    print("\nAverage cost indices (basis points):")
    for ft, row in idx_stats.iterrows():
        print(
            f"  {ft}: total {_bp(row['cost_index'])}, per-kg {_bp(row['cost_per_kg_index'])}, per-drop {_bp(row['cost_per_drop_index'])}"
        )
 
    # Find best performing configurations
    if len(df[df['fleet_type'] == 'MCV']) > 0 and len(df[df['fleet_type'] == 'SCV']) > 0:
        print(f"\nBest MCV configurations (lowest cost per kg):")
        best_mcv = df[df['fleet_type'] == 'MCV'].nsmallest(5, 'cost_per_kg')[
            ['day_id', 'alpha', 'C', 'cost_per_kg', 'solver_cost', 'alpha_surcharge_pct', 'c_pct_scv', 'cost_per_kg_index']
        ]
        for _, row in best_mcv.iterrows():
            print(
                f"Day {row['day_id']}: α={row['alpha']:.2f} ({row['alpha_surcharge_pct']:.0f}% surcharge), "
                f"C={row['C']:.0f} ({row['c_pct_scv']:.1f}% of SCV), "
                f"cost/kg={row['cost_per_kg']:.3f} ({_bp(row['cost_per_kg_index'])})"
            )
        
        print(f"\nSCV baseline (cost per kg):")
        scv_stats = df[df['fleet_type'] == 'SCV'][['cost_per_kg', 'solver_cost']].agg(['mean', 'std'])
        print(scv_stats)

def compute_stats(df, cost_col='solver_cost'):
    mcv = df[df['fleet_type'] == 'MCV'].copy()
    scv = df[df['fleet_type'] == 'SCV'][['day_id', cost_col]].rename(columns={cost_col: f'{cost_col}_scv'}).set_index('day_id')
    
    def group_stats(g):
        costs = g.merge(scv, left_on='day_id', right_index=True)
        diff = costs[f'{cost_col}_scv'] - costs[cost_col]
        rel_diff = diff / costs[f'{cost_col}_scv']
        win_rate = (diff > 0).mean()
        avg_pct = rel_diff.mean() * 100
        avg_pct_wins = rel_diff[diff > 0].mean() * 100 if any(diff > 0) else 0
        avg_pct_losses = rel_diff[diff <= 0].mean() * 100 if any(diff <= 0) else 0
        return pd.Series({
            'win_rate': win_rate,
            'avg_pct_savings': avg_pct,
            'avg_pct_savings_wins': avg_pct_wins,
            'avg_pct_savings_losses': avg_pct_losses,
            'num_days': len(g)
        })
    
    stats = mcv.groupby(['alpha_surcharge_pct', 'c_pct_scv']).apply(group_stats, include_groups=False).reset_index()
    return stats

def get_savings_df(df, cost_col='solver_cost'):
    mcv = df[df['fleet_type'] == 'MCV'].copy()
    scv = df[df['fleet_type'] == 'SCV'][['day_id', cost_col]].rename(columns={cost_col: f'{cost_col}_scv'}).set_index('day_id')
    merged = mcv.merge(scv, left_on='day_id', right_index=True)
    merged['pct_savings'] = (merged[f'{cost_col}_scv'] - merged[cost_col]) / merged[f'{cost_col}_scv'] * 100
    return merged

def plot_win_rate_savings_map(stats, figs_dir, metric_name='total_cost'):
    # Pivot tables for colour (average %-savings) and win-rate, rounding for readability
    pivot_pct = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').round(1)
    pivot_wr = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='win_rate')

    # Build a nicer annotation: "12%\n23/30"
    days = int(stats['num_days'].max())  # assume each cell is evaluated on the same #days
    annot = pd.DataFrame(index=pivot_pct.index, columns=pivot_pct.columns, dtype=object)
    for a in pivot_pct.index:
        for c in pivot_pct.columns:
            val = pivot_pct.loc[a, c]
            if pd.isna(val):
                annot.loc[a, c] = ""
            else:
                wins = int(round(pivot_wr.loc[a, c] * days))
                annot.loc[a, c] = f"{val:.0f}%\n{wins}/{days}"

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_pct,
        annot=annot,
        fmt="",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Average %-Savings vs SCV'}
    )
    plt.title(
        f"MCV Advantage ({metric_name}): Avg %-Savings (colour)  |  Wins/Days (text)"
    )
    plt.ylabel("MCV Surcharge % (from α)")
    plt.xlabel("C as % of SCV fixed cost")
    plt.tight_layout()
    plt.savefig(figs_dir / f'win_rate_savings_map_{metric_name}.png')
    plt.close()

def plot_savings_distributions(savings_df, figs_dir, metric_name='total_cost'):
    g = sns.FacetGrid(savings_df, row='alpha_surcharge_pct', col='c_pct_scv', margin_titles=True, sharex=True)
    g.map(sns.boxplot, 'pct_savings')
    g.set_axis_labels(" %-Savings vs SCV", "")
    g.fig.suptitle(f"Savings Distributions ({metric_name})", y=1.02)
    g.add_legend()
    plt.tight_layout()
    plt.savefig(figs_dir / f'savings_distributions_{metric_name}.png')
    plt.close()

def plot_break_even(stats, figs_dir, threshold=0.5, metric_name='total_cost'):
    break_evens = []
    for c in sorted(stats['c_pct_scv'].unique()):
        c_data = stats[stats['c_pct_scv'] == c].sort_values('alpha_surcharge_pct')
        eligible = c_data[c_data['win_rate'] >= threshold]
        max_alpha = eligible['alpha_surcharge_pct'].max() if not eligible.empty else float('-inf')
        break_evens.append({'c_pct_scv': c, 'break_even_alpha': max_alpha})
    
    be_df = pd.DataFrame(break_evens)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=be_df, x='c_pct_scv', y='break_even_alpha', marker='o')

    # Annotate each point with the %-surcharge value
    for _, row in be_df.iterrows():
        if row['break_even_alpha'] > float('-inf'):
            plt.text(
                row['c_pct_scv'],
                row['break_even_alpha'] + 2,  # slight offset
                f"{row['break_even_alpha']:.0f}%",
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.title(
        f'Maximum MCV Fixed-Cost Premium Allowable for {int(threshold*100)}% Win-Rate ({metric_name})'
    )
    plt.xlabel('Compartment set-up cost  C  (% of SCV fixed cost)')
    plt.ylabel('Max surcharge on MCV fixed cost (%)')
    finite_vals = be_df['break_even_alpha'].replace(-np.inf, np.nan).dropna()
    if not finite_vals.empty:
        plt.ylim(0, finite_vals.max() * 1.1)
    else:
        plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / f'break_even_{int(threshold*100)}_{metric_name}.png')
    plt.close()

def plot_frontier(stats, figs_dir, metric_name='total_cost'):
    stats['p'] = stats['win_rate']
    stats['m'] = stats['avg_pct_savings_wins']
    stats['p_m'] = stats['p'] * stats['m']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stats, x='p', y='p_m', hue='c_pct_scv', size='alpha_surcharge_pct', sizes=(20, 200))
    plt.title(f'Frontier: Win Prob vs Prob-Weighted Magnitude ({metric_name})')
    plt.xlabel('Win Probability')
    plt.ylabel('Expected %-Savings (when winning)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / f'frontier_{metric_name}.png')
    plt.close()

def plot_daily_leaderboard(savings_df, figs_dir, metric_name='total_cost'):
    # Optionally exclude the “free-lunch” baseline where both surcharge and C are zero
    filtered = savings_df[~((savings_df['alpha_surcharge_pct'] == 0) & (savings_df['c_pct_scv'] == 0))]
    if filtered.empty:  # fall back if data filtered away completely
        filtered = savings_df.copy()

    best_per_day = filtered.loc[filtered.groupby('day_id')['pct_savings'].idxmax()]
    best_per_day['param_pair'] = 'α=' + best_per_day['alpha_surcharge_pct'].astype(str) + '%, C=' + best_per_day['c_pct_scv'].astype(str) + '%'
    best_per_day = best_per_day.sort_values('total_demand_kg')
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=best_per_day, x='day_id', y='pct_savings', hue='param_pair', dodge=False)
    plt.title(f'Best MCV %-Savings per Day ({metric_name})')
    plt.xlabel('Day ID (sorted by total demand kg)')
    plt.ylabel('%-Savings vs SCV')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(figs_dir / f'daily_leaderboard_{metric_name}.png')
    plt.close()

def main():
    """Run partial analysis."""
    print("="*60)
    print("ALPHA ANALYSIS - PARTIAL RESULTS")
    print("="*60)
    
    print("Loading partial results...")
    df = load_partial_results()
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    print(f"\nAnalyzing {len(df)} results...")
    
    # Summary statistics
    summary_stats(df)
    
    # Create plots
    print(f"\nGenerating plots (saving to {FIGS_DIR})...")
    plot_cost_trends(df)
    plot_index_trends(df)
    plot_fleet_comparison(df)
    coverage_analysis(df)
    
    # New metrics 1-7
    for cost_col, metric_name in [
        ('solver_cost', 'total_cost'),
        ('cost_per_kg', 'per_kg'),
        ('cost_per_drop', 'per_drop')
    ]:
        print(f"Computing metrics for {metric_name}")
        stats = compute_stats(df, cost_col)
        savings_df = get_savings_df(df, cost_col)
        
        # Metrics 1-2
        plot_win_rate_savings_map(stats, FIGS_DIR, metric_name)
        
        # Metric 3
        plot_savings_distributions(savings_df, FIGS_DIR, metric_name)
        
        # Metric 4 (using 50% and 90% thresholds)
        plot_break_even(stats, FIGS_DIR, 0.5, metric_name)
        plot_break_even(stats, FIGS_DIR, 0.9, metric_name)
        
        # Metric 5
        plot_frontier(stats, FIGS_DIR, metric_name)
        
        # Metric 7 (leaderboard is per-day, fits here)
        plot_daily_leaderboard(savings_df, FIGS_DIR, metric_name)
    
    # Save partial summary
    summary_path = Path("results/partial_summary.parquet")
    summary_path.parent.mkdir(exist_ok=True)
    df.to_parquet(summary_path)
    print(f"\nSaved partial summary to {summary_path}")
    print(f"Figures saved to {FIGS_DIR}/")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 