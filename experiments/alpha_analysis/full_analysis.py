"""
Full analysis script for alpha grid search results.
Implements the curated menu of analyses and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
from fleetmix.config import load_fleetmix_params

BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config.yaml")

RESULTS_RAW = Path("results_raw_v4")
FIGS_DIR = Path("results/full_figs_v4")
FIGS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("results/tables_v4")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _bp(value: float) -> str:
    """Format a ratio (1 = 100%) as basis points string."""
    return f"{value * 10000:.0f} bp"

def load_results():
    """Load all existing JSON results with additional computations."""
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
        scv_mean_vehicles = scv_df['total_vehicles'].mean()
        scv_mean_fixed = scv_df['total_fixed_cost'].mean()
        scv_mean_variable = scv_df['total_variable_cost'].mean()
        
        df['cost_index'] = df['solver_cost'] / scv_mean_cost
        df['cost_per_kg_index'] = df['cost_per_kg'] / scv_mean_kg
        df['cost_per_drop_index'] = df['cost_per_drop'] / scv_mean_drop
        df['vehicles_index'] = df['total_vehicles'] / scv_mean_vehicles
        df['fixed_index'] = df['total_fixed_cost'] / scv_mean_fixed
        df['variable_index'] = df['total_variable_cost'] / scv_mean_variable
    
    # Compute load factor
    def compute_total_capacity(row):
        vehicles_used = row['vehicles_used']
        fleet_type = row['fleet_type']
        if fleet_type == 'SCV':
            # Assuming all SCV have capacity 2700 based on templates
            return sum(vehicles_used.values()) * 2700
        else:
            # MCV capacities from default
            capacities = {'A': 2700, 'B': 3300, 'C': 4500}
            total = 0
            for vt, count in vehicles_used.items():
                total += count * capacities.get(vt, 2700)  # Default to 2700 if unknown
            return total
    
    df['total_capacity'] = df.apply(compute_total_capacity, axis=1)
    df['load_factor'] = df['total_demand_kg'] / df['total_capacity'].replace(0, np.nan)
    
    # Compute driver hours index
    if len(scv_df) > 0:
        scv_mean_driver_hours = scv_df['total_route_time_hours'].mean()
        df['driver_hours_index'] = df['total_route_time_hours'] / scv_mean_driver_hours
    
    return df

# Reused from partial_analysis (adjusted for new metrics)
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
        median_pct_wins = rel_diff[diff > 0].median() * 100 if any(diff > 0) else 0
        avg_pct_losses = rel_diff[diff <= 0].mean() * 100 if any(diff <= 0) else 0
        # New derived metrics
        ev_saving = win_rate * avg_pct_wins  # Expected value of savings
        conditional_loss = (1 - win_rate) * abs(avg_pct_losses)
        nbi = ev_saving - conditional_loss
        return pd.Series({
            'win_rate': win_rate,
            'avg_pct_savings': avg_pct,
            'avg_pct_savings_wins': avg_pct_wins,
            'median_pct_savings_wins': median_pct_wins,
            'avg_pct_savings_losses': avg_pct_losses,
            'ev_saving': ev_saving,
            'conditional_loss': conditional_loss,
            'nbi': nbi,
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

# 1. Economic “Sweet-Spot” Mapping
def plot_economic_sweet_spot(df, figs_dir):
    stats = compute_stats(df)
    pivot = stats.pivot(index='alpha_surcharge_pct', columns='c_pct_scv', values='avg_pct_savings').fillna(0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Average % Cost Savings (positive = MCV better)'}
    )
    
    # Add zero contour
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    plt.contour(X, Y, Z, levels=[0], colors='black', linestyles='solid', linewidths=2)
    
    plt.title('Economic Sweet-Spot: Average % Cost Savings (MCV vs SCV)')
    plt.xlabel('Setup Cost C (% of SCV cap-ex)')
    plt.ylabel('Vehicle Surcharge α (%)')
    plt.tight_layout()
    plt.savefig(figs_dir / 'economic_sweet_spot.png', dpi=150, bbox_inches='tight')
    plt.close()

# TODO: Cost-Structure Decomposition

# 3. Probability-of-Superiority Analysis
def plot_probability_superiority(df, figs_dir):
    stats = compute_stats(df)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=stats,
        x='c_pct_scv',
        y='alpha_surcharge_pct',
        size='win_rate',
        sizes=(20, 500),
        hue='median_pct_savings_wins',
        palette='RdYlGn',
        hue_norm=(0, 20),  # Adjust based on data range
        alpha=0.7
    )
    plt.title('Probability of MCV Superiority: Size = Win Probability, Color = Median % Savings when Winning')
    plt.xlabel('Setup Cost C (% of SCV cap-ex)')
    plt.ylabel('Vehicle Surcharge α (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / 'probability_superiority.png', dpi=150, bbox_inches='tight')
    plt.close()

# TODO ???  Normalised Performance Distributions
    mcv_df = df[df['fleet_type'] == 'MCV']
    selected_alphas = [0, 25, 50, 75, 100]  # Example levels
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost per kg
    sns.violinplot(
        data=mcv_df[mcv_df['alpha_surcharge_pct'].isin(selected_alphas)],
        x='alpha_surcharge_pct',
        y='cost_per_kg_index',
        ax=axs[0],
        inner='box'
    )
    axs[0].axhline(1, color='red', linestyle='--', label='SCV Baseline')
    axs[0].set_title('Cost per kg Ratio (MCV / SCV)')
    axs[0].set_xlabel('Vehicle Surcharge α (%)')
    axs[0].set_ylabel('Cost Index (1.0 = SCV)')
    axs[0].legend()
    
    # Cost per drop
    sns.violinplot(
        data=mcv_df[mcv_df['alpha_surcharge_pct'].isin(selected_alphas)],
        x='alpha_surcharge_pct',
        y='cost_per_drop_index',
        ax=axs[1],
        inner='box'
    )
    axs[1].axhline(1, color='red', linestyle='--', label='SCV Baseline')
    axs[1].set_title('Cost per Drop Ratio (MCV / SCV)')
    axs[1].set_xlabel('Vehicle Surcharge α (%)')
    axs[1].set_ylabel('Cost Index (1.0 = SCV)')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(figs_dir / 'normalized_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

# 5. Operational KPIs Beyond Cost
def plot_operational_kpis(df, figs_dir):
    mcv_df = df[df['fleet_type'] == 'MCV']
    scv_df = df[df['fleet_type'] == 'SCV']
    
    # Vehicles used
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=mcv_df,
        x='alpha_surcharge_pct',
        y='total_vehicles',
        hue='c_pct_scv',
        errorbar='sd',
        marker='o'
    )
    if not scv_df.empty:
        scv_mean = scv_df['total_vehicles'].mean()
        plt.axhline(scv_mean, color='red', linestyle='--', label=f'SCV Mean: {scv_mean:.1f}')
    plt.title('Average Vehicles Used vs Surcharge (Ribbon = SD)')
    plt.xlabel('Vehicle Surcharge α (%)')
    plt.ylabel('Number of Vehicles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / 'vehicles_used.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # TODO Average load factor
    
    # Driver-hours index
    fig, ax = plt.subplots(figsize=(8, 6))
    means = df.groupby('fleet_type')['total_route_time_hours'].mean()
    errors = df.groupby('fleet_type')['total_route_time_hours'].std()
    means.plot(kind='bar', yerr=errors, ax=ax, capsize=4, color=['blue', 'green'])
    ax.set_title('Average Driver-Hours per Fleet Type')
    ax.set_ylabel('Hours')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figs_dir / 'driver_hours.png', dpi=150, bbox_inches='tight')
    plt.close()

# 6.TODO: ??? Sensitivity / Robustness Dashboard

# 7. Computational Footprint vs Instance Size
def plot_computational_footprint(df, figs_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='num_customers',
        y='solver_runtime_sec',
        hue='fleet_type',
        style='fleet_type',
        s=100,
        alpha=0.7
    )
    plt.yscale('log')
    plt.title('Solve Time vs Instance Size')
    plt.xlabel('Number of Customers')
    plt.ylabel('Solve Time (seconds, log scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / 'computational_footprint.png', dpi=150, bbox_inches='tight')
    plt.close()

# 8. Story-telling Tables
def generate_tables(df, tables_dir):
    stats = compute_stats(df)
    
    # Table A1 – Key thresholds
    break_evens = []
    for c in sorted(stats['c_pct_scv'].unique()):
        c_data = stats[stats['c_pct_scv'] == c].sort_values('alpha_surcharge_pct')
        eligible = c_data[c_data['win_rate'] >= 0.5]
        max_alpha = eligible['alpha_surcharge_pct'].max() if not eligible.empty else np.nan
        # Median saving at α=1.0 (0% surcharge)
        at_zero = c_data[c_data['alpha_surcharge_pct'] == 0]
        median_saving = at_zero['median_pct_savings_wins'].iloc[0] if not at_zero.empty else np.nan
        break_evens.append({'C (% of SCV)': c, 'Break-even α (%)': max_alpha, 'Median saving at α=1.0': median_saving})
    
    table_a1 = pd.DataFrame(break_evens)
    table_a1.to_markdown(tables_dir / 'table_a1.md', index=False)
    
    # Table A2 – 95% confidence intervals for main KPIs at flagship settings
    flagship = df[(df['alpha_surcharge_pct'] == 20) & (df['c_pct_scv'] == 10) & (df['fleet_type'] == 'MCV')]
    if not flagship.empty:
        kpis = ['solver_cost', 'cost_per_kg', 'cost_per_drop', 'total_vehicles', 'load_factor']
        ci_data = []
        for kpi in kpis:
            mean = flagship[kpi].mean()
            std = flagship[kpi].std()
            n = len(flagship)
            ci_low = mean - 1.96 * std / np.sqrt(n)
            ci_high = mean + 1.96 * std / np.sqrt(n)
            ci_data.append({'KPI': kpi, 'Mean': mean, '95% CI': f'[{ci_low:.2f}, {ci_high:.2f}]'})
        table_a2 = pd.DataFrame(ci_data)
        table_a2.to_markdown(tables_dir / 'table_a2.md', index=False)

# TODO: Fleet Capital Efficiency Frontier The Fleet Transformation Frontier: Trading Capital Premium for Fleet Efficiency

# TODO. Vehicle Mix Composition Heat Map


# TODO. ???? Fleet Downsizing Economics   Translates fleet reduction into economic value."""


# 12. Urban Congestion Impact
def plot_urban_impact(df, figs_dir):
    """Shows the societal benefit of fewer vehicles on roads."""
    stats_by_config = []
    
    # Calculate metrics for each configuration
    for fleet_type in df['fleet_type'].unique():
        fleet_df = df[df['fleet_type'] == fleet_type]
        
        if fleet_type == 'SCV':
            # Single SCV configuration
            vehicles_per_100_customers = fleet_df['total_vehicles'].sum() / fleet_df['num_customers'].sum() * 100
            avg_route_hours = fleet_df['total_route_time_hours'].mean()
            
            stats_by_config.append({
                'config': 'SCV',
                'fleet_type': fleet_type,
                'alpha': 0,
                'c': 0,
                'vehicles_per_100_customers': vehicles_per_100_customers,
                'vehicle_hours_on_road': avg_route_hours
            })
        else:
            # Multiple MCV configurations
            for (alpha, c), group in fleet_df.groupby(['alpha_surcharge_pct', 'c_pct_scv']):
                vehicles_per_100_customers = group['total_vehicles'].sum() / group['num_customers'].sum() * 100
                avg_route_hours = group['total_route_time_hours'].mean()
                
                stats_by_config.append({
                    'config': f'MCV (α={alpha}%, C={c}%)',
                    'fleet_type': fleet_type,
                    'alpha': alpha,
                    'c': c,
                    'vehicles_per_100_customers': vehicles_per_100_customers,
                    'vehicle_hours_on_road': avg_route_hours
                })
    
    config_df = pd.DataFrame(stats_by_config)
    scv_baseline = config_df[config_df['fleet_type'] == 'SCV']['vehicles_per_100_customers'].iloc[0]
    
    # Focus on a few interesting configurations
    interesting_configs = [
        ('SCV', 0, 0),
        ('MCV', 0, 10),
        ('MCV', 25, 10),
        ('MCV', 50, 20)
    ]
    
    filtered = config_df[config_df[['fleet_type', 'alpha', 'c']].apply(tuple, axis=1).isin(interesting_configs)]
    filtered = filtered.sort_values('vehicles_per_100_customers', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of vehicles per 100 customers
    bars = ax1.bar(range(len(filtered)), filtered['vehicles_per_100_customers'])
    ax1.set_xticks(range(len(filtered)))
    ax1.set_xticklabels(filtered['config'], rotation=45, ha='right')
    ax1.set_ylabel('Vehicles per 100 Customers')
    ax1.set_title('Fleet Density: Fewer Vehicles, Less Congestion')
    ax1.axhline(scv_baseline, color='red', linestyle='--', label='SCV Baseline')
    
    # Color bars based on fleet type
    for i, (idx, row) in enumerate(filtered.iterrows()):
        if row['fleet_type'] == 'SCV':
            bars[i].set_color('lightcoral')
        else:
            bars[i].set_color('lightgreen')
            # Add reduction percentage
            reduction = (1 - row['vehicles_per_100_customers'] / scv_baseline) * 100
            ax1.text(i, row['vehicles_per_100_customers'] + 0.5, 
                    f'-{reduction:.0f}%', ha='center', fontsize=10, color='darkgreen', weight='bold')
    
    ax1.legend()
    
    # Scatter plot showing efficiency
    colors = ['red' if ft == 'SCV' else 'green' for ft in filtered['fleet_type']]
    ax2.scatter(filtered['vehicles_per_100_customers'], 
               filtered['vehicle_hours_on_road'],
               s=200, alpha=0.6, c=colors)
    
    for idx, row in filtered.iterrows():
        ax2.annotate(row['config'], 
                    (row['vehicles_per_100_customers'], row['vehicle_hours_on_road']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Vehicles per 100 Customers')
    ax2.set_ylabel('Average Vehicle-Hours on Road')
    ax2.set_title('The Double Dividend: Fewer Vehicles AND Shorter Routes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figs_dir / 'urban_impact.png', dpi=150, bbox_inches='tight')
    plt.close()

# 13. Scale-Dependent Fleet Benefits
def plot_scale_benefits(df, figs_dir):
    """Shows how MCV benefits change with operation scale."""
    # Create customer bins
    df['customer_bin'] = pd.qcut(df['num_customers'], q=5, duplicates='drop')
    
    scale_analysis = []
    
    for bin_label in df['customer_bin'].unique():
        bin_data = df[df['customer_bin'] == bin_label]
        
        # Average number of customers in this bin
        avg_customers = bin_data['num_customers'].mean()
        
        # SCV baseline for this scale
        scv_vehicles = bin_data[bin_data['fleet_type'] == 'SCV']['total_vehicles'].mean()
        
        if pd.isna(scv_vehicles):
            continue
            
        # Key MCV scenarios
        for (alpha, c) in [(0, 10), (25, 10), (50, 20)]:
            mcv_data = bin_data[(bin_data['fleet_type'] == 'MCV') & 
                               (bin_data['alpha_surcharge_pct'] == alpha) & 
                               (bin_data['c_pct_scv'] == c)]
            
            if not mcv_data.empty:
                mcv_vehicles = mcv_data['total_vehicles'].mean()
                fleet_reduction = (1 - mcv_vehicles / scv_vehicles) * 100
                
                scale_analysis.append({
                    'customer_range': str(bin_label),
                    'avg_customers': avg_customers,
                    'scenario': f'α={alpha}%, C={c}%',
                    'fleet_reduction_pct': fleet_reduction
                })
    
    scale_df = pd.DataFrame(scale_analysis)
    
    if not scale_df.empty:
        plt.figure(figsize=(12, 7))
        
        # Plot each scenario
        for scenario in scale_df['scenario'].unique():
            scenario_data = scale_df[scale_df['scenario'] == scenario].sort_values('avg_customers')
            plt.plot(scenario_data['avg_customers'], 
                    scenario_data['fleet_reduction_pct'],
                    marker='o', markersize=10, linewidth=2.5,
                    label=scenario)
        
        plt.xlabel('Operation Scale (Number of Customers)')
        plt.ylabel('Fleet Size Reduction (%)')
        plt.title('The Scale Dividend: Larger Operations Benefit More from Fleet Flexibility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add insight annotation
        plt.text(0.02, 0.02, 
                'Insight: MCV benefits compound with scale—perfect for growing logistics operations',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(figs_dir / 'scale_benefits.png', dpi=150, bbox_inches='tight')
        plt.close()

# 14. Win–Loss Trade-off Scatter (Avg % Saving vs Win Probability)

def plot_win_loss_tradeoff(df, figs_dir):
    """Each point = (α, C).  X-axis: average % saving (MCV vs SCV),
    Y-axis: win probability.  Helps visualise frequency vs magnitude.
    """
    stats = compute_stats(df)

    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=stats,
        x="avg_pct_savings",
        y="win_rate",
        hue="c_pct_scv",
        style="alpha_surcharge_pct",
        s=150,
        palette="viridis",
        alpha=0.85,
        edgecolor="black",
    )

    # Reference lines
    plt.axvline(0, color="grey", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="grey", linestyle="--", linewidth=1)

    plt.xlabel("Average % Saving (positive = MCV better)")
    plt.ylabel("Win Probability")
    plt.title("Win–Loss Trade-off: How Often vs How Much Does MCV Win?")
    plt.legend(title="C (% of SCV)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "win_loss_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


# 15. Savings Distribution ECDFs for Selected Configurations

def plot_savings_ecdf(df, figs_dir, selected_configs=None):
    """Plot cumulative distribution (ECDF) of % savings for a handful of (α, C)
    settings to show variability and downside risk.

    Parameters
    ----------
    selected_configs : list of tuples, optional
        Each tuple = (alpha_surcharge_pct, c_pct_scv). Defaults to three
        illustrative configs.
    """
    if selected_configs is None:
        selected_configs = [(0, 0), (0, 10), (25, 10), (50, 20)]

    savings_df = get_savings_df(df)

    plt.figure(figsize=(10, 7))
    for alpha_pct, c_pct in selected_configs:
        subset = savings_df[(savings_df["alpha_surcharge_pct"] == alpha_pct) &
                            (savings_df["c_pct_scv"] == c_pct)]
        if subset.empty:
            continue
        label = f"α={alpha_pct}%, C={c_pct}%"
        sns.ecdfplot(subset["pct_savings"], label=label, linewidth=2)

    plt.axvline(0, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("% Saving (positive = MCV cheaper)")
    plt.ylabel("Cumulative Probability")
    plt.title("Distribution of Daily Savings Across Configurations")
    plt.legend(title="Configuration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "savings_ecdf.png", dpi=150, bbox_inches="tight")
    plt.close()

# 16. Viability Frontier based on Net Benefit Index (NBI)

def plot_viability_frontier(df, figs_dir):
    """Heatmap of NBI with contour lines of EV-Saving, highlighting the region
    where overall expected benefit is positive.
    """
    stats = compute_stats(df)
    # Pivot for NBI heatmap
    nbi_pivot = stats.pivot(index="alpha_surcharge_pct", columns="c_pct_scv", values="nbi").fillna(0)
    ev_pivot = stats.pivot(index="alpha_surcharge_pct", columns="c_pct_scv", values="ev_saving").fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        nbi_pivot,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Net Benefit Index (bp)"}
    )

    # Contour of EV-Saving (dashed)
    X, Y = np.meshgrid(nbi_pivot.columns, nbi_pivot.index)
    Z = ev_pivot.values
    CS = plt.contour(X, Y, Z, levels=[0, 5, 10, 15, 20], colors="black", linestyles="dashed")
    plt.clabel(CS, inline=True, fontsize=8, fmt="EV %.0f%%")

    plt.title("Viability Frontier: Where MCV Beats SCV on Expected Value")
    plt.xlabel("Setup Cost C (% of SCV cap-ex)")
    plt.ylabel("Vehicle Surcharge α (%)")
    plt.tight_layout()
    plt.savefig(figs_dir / "viability_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()


# 17. Box-and-Whisker Grid of Daily Savings

def plot_boxplot_grid(df, figs_dir):
    """Mini boxplots of daily % savings for each (α, C) cell to visualise
    distribution and outliers."""
    savings_df = get_savings_df(df)
    alphas = sorted(savings_df["alpha_surcharge_pct"].unique())
    cs = sorted(savings_df["c_pct_scv"].unique())

    n_rows = len(alphas)
    n_cols = len(cs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharey=True)

    for i, a in enumerate(alphas):
        for j, c in enumerate(cs):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            subset = savings_df[(savings_df["alpha_surcharge_pct"] == a) & (savings_df["c_pct_scv"] == c)]
            if subset.empty:
                ax.axis("off")
                continue
            sns.boxplot(y=subset["pct_savings"], ax=ax, color="lightsteelblue", fliersize=1, width=0.5)
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f"α={a}%")
            else:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(f"C={c}%")
            ax.set_xlabel("")
            ax.set_ylim(-40, 40)

    plt.suptitle("Distribution of Daily Savings per Configuration", y=1.02)
    plt.tight_layout()
    plt.savefig(figs_dir / "savings_boxplot_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


# Functions copied from partial_analysis.py

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

def plot_daily_leaderboard(savings_df, figs_dir, metric_name='total_cost'):
    # Optionally exclude the "free-lunch" baseline where both surcharge and C are zero
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

# TODO: compartment activation 

def main():
    print("="*60)
    print("ALPHA ANALYSIS - FULL RESULTS")
    print("="*60)
    
    print("Loading results...")
    df = load_results()
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    print(f"\nAnalyzing {len(df)} results...")
    
    # Generate plots
    print(f"\nGenerating plots (saving to {FIGS_DIR})...")
    plot_economic_sweet_spot(df, FIGS_DIR)
    plot_probability_superiority(df, FIGS_DIR)
    plot_operational_kpis(df, FIGS_DIR)
    plot_computational_footprint(df, FIGS_DIR)
    
    # New plots introduced v4.1
    plot_win_loss_tradeoff(df, FIGS_DIR)
    plot_savings_ecdf(df, FIGS_DIR)
    
    # Viability frontier & distribution visuals
    plot_viability_frontier(df, FIGS_DIR)
    plot_boxplot_grid(df, FIGS_DIR)
    
    # New vehicle fleet analyses
    print("\nGenerating vehicle fleet analyses...")
    plot_urban_impact(df, FIGS_DIR)
    plot_scale_benefits(df, FIGS_DIR)
    
    # Generate additional analysis plots from partial_analysis
    print("\nGenerating additional analysis plots...")
    for cost_col, metric_name in [
        ('solver_cost', 'total_cost'),
        ('cost_per_kg', 'per_kg'),
        ('cost_per_drop', 'per_drop')
    ]:
        print(f"Computing metrics for {metric_name}")
        stats = compute_stats(df, cost_col)
        savings_df = get_savings_df(df, cost_col)
        
        # Win rate savings map
        plot_win_rate_savings_map(stats, FIGS_DIR, metric_name)
        
        # Break-even analysis (using 50% and 90% thresholds)
        plot_break_even(stats, FIGS_DIR, 0.5, metric_name)
        plot_break_even(stats, FIGS_DIR, 0.9, metric_name)
        
        # Daily leaderboard
        plot_daily_leaderboard(savings_df, FIGS_DIR, metric_name)
    
    # Generate tables
    print(f"\nGenerating tables (saving to {TABLES_DIR})...")
    generate_tables(df, TABLES_DIR)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()