"""
Post-processing and plots for alpha analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from pathlib import Path

SUMMARY_PATH = Path("results/summary_v3.parquet")
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_parquet(SUMMARY_PATH)
    
    # Cost curves vs alpha
    plt.figure(figsize=(10, 6))
    scv_mean = df[df['fleet_type'] == 'SCV']['solver_cost'].mean()
    plt.axhline(scv_mean, color='r', label='SCV')
    sns.lineplot(data=df[df['fleet_type'] == 'MCV'], x='alpha', y='solver_cost', hue='C', ci='sd')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Solver Cost')
    plt.legend()
    plt.savefig(FIGS_DIR / 'curve.png')
    
    # Breakeven alpha*
    breakevens = []
    for day, day_df in df.groupby('day_id'):
        scv_cost = day_df[day_df['fleet_type'] == 'SCV']['solver_cost'].iloc[0]
        mcv_df = day_df[day_df['fleet_type'] == 'MCV']
        for c in mcv_df['C'].unique():
            c_df = mcv_df[mcv_df['C'] == c].sort_values('alpha')
            if len(c_df) < 2: continue
            interp = interp1d(c_df['solver_cost'], c_df['alpha'], kind='linear', fill_value="extrapolate")
            alpha_star = interp(scv_cost)
            breakevens.append({'day_id': day, 'C': c, 'alpha_star': alpha_star})
    break_df = pd.DataFrame(breakevens)
    break_df.to_csv(FIGS_DIR / 'breakeven_table.csv', index=False)
    
    # Violin plots
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='fleet_type', y='cost_per_kg', hue='fleet_type')
    plt.savefig(FIGS_DIR / 'violin_normalized.png')

if __name__ == "__main__":
    main() 