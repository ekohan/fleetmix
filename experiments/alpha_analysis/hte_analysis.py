import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path
from typing import List
import math

# Optional: set random seed for reproducibility in CV
RNG = np.random.default_rng(42)

from experiments.alpha_analysis.dataset_characterization import _aggregate_daily_stats  # Reuse existing characterization
from experiments.alpha_analysis.full_analysis import load_results  # Import proper loader

# Paths
RESULTS_DIR = Path("results/alpha_analysis")
CHAR_PATH = Path("results/demand_characterization/daily_summary.csv")
OUTPUT_DIR = Path("results/hte_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NEW CONSTANT: toggle LaTeX export
EXPORT_LATEX = True

# NEW UTILITY FUNCTIONS

def _center_scale(df: pd.DataFrame, cols: List[str], center_only: bool = True) -> pd.DataFrame:
    """Return a copy of *df* with extra columns `{col}_c` that are centred (and optionally scaled).

    Parameters
    ----------
    cols : list of column names to centre/scale.
    center_only : if True, subtract mean; if False, also divide by std.
    """
    df_cs = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std() if not center_only else 1.0
        df_cs[f"{col}_c"] = (df[col] - mean) / std
    return df_cs


def _save_latex_summary(model, filename: str):
    if EXPORT_LATEX:
        tex_path = OUTPUT_DIR / filename
        tex_path.write_text(model.summary().as_latex())
        print(f"LaTeX summary written to {tex_path}")


def _cross_val_rmse(df: pd.DataFrame, formula: str, k: int = 5) -> float:
    """Compute k-fold cross-validated RMSE using statsmodels formulas without external deps."""
    indices = np.arange(len(df))
    RNG.shuffle(indices)
    fold_sizes = np.full(k, len(indices) // k)
    fold_sizes[: len(indices) % k] += 1  # distribute remainder
    current = 0
    rmses = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        model = smf.ols(formula, data=train_df).fit()
        preds = model.predict(test_df)
        rmse = math.sqrt(np.mean((test_df["cost_diff"] - preds) ** 2))
        rmses.append(rmse)
        current = stop
    return float(np.mean(rmses))


def load_and_prepare_data():
    raw_df = load_results()
    scv = raw_df[raw_df['fleet_type'] == 'SCV'][['day_id', 'solver_cost']].rename(columns={'solver_cost': 'scv_cost'}).set_index('day_id')
    mcv = raw_df[raw_df['fleet_type'] == 'MCV']
    df = mcv.merge(scv, left_on='day_id', right_index=True)
    df['cost_diff'] = df['scv_cost'] - df['solver_cost']
    return df

def fit_rsm(df):
    model = smf.ols('cost_diff ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C', data=df).fit()
    print(model.summary())
    return model

def plot_response_surface(df, model, suffix=''):
    # Grid for surface
    alpha_grid = np.linspace(df['alpha'].min(), df['alpha'].max(), 100)
    c_grid = np.linspace(df['C'].min(), df['C'].max(), 100)
    A, CC = np.meshgrid(alpha_grid, c_grid)
    pred_df = pd.DataFrame({'alpha': A.ravel(), 'C': CC.ravel()})
    pred = model.predict(pred_df).values.reshape(A.shape)

    # 3D Surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, CC, pred, cmap='viridis')
    ax.set_xlabel('Fixed-cost multiplier α (-)')
    ax.set_ylabel('Compartment setup cost C')
    ax.set_zlabel('ΔCost (SCV − MCV)')
    ax.view_init(elev=30, azim=-135)
    fig.tight_layout(pad=0.3)
    plt.savefig(OUTPUT_DIR / f'rsm_3d{suffix}.png', dpi=300)
    plt.close(fig)

    # Contour
    fig2 = plt.figure(figsize=(8, 6))
    contour = plt.contourf(A, CC, pred, levels=20, cmap='RdBu')
    plt.colorbar(contour, label='ΔCost (SCV − MCV)')
    plt.xlabel('Fixed-cost multiplier α (-)')
    plt.ylabel('Compartment setup cost C')
    plt.contour(A, CC, pred, levels=[0], colors='black', linestyles='dashed')
    fig2.tight_layout(pad=0.3)
    plt.savefig(OUTPUT_DIR / f'rsm_contour{suffix}.png', dpi=300)
    plt.close(fig2)

def demand_moderation(df, char_df):
    merged = df.merge(char_df, on='day_id')
    mod_model = smf.ols('cost_diff ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C + demand_total_kg + demand_customers + alpha:demand_total_kg + C:demand_total_kg + alpha:demand_customers + C:demand_customers', data=merged).fit()
    print(mod_model.summary())
    return mod_model

def stratified_rsm(df, char_df):
    merged = df.merge(char_df, on='day_id')
    merged['demand_level'] = pd.qcut(merged['demand_total_kg'], 3, labels=['Low', 'Medium', 'High'])
    for level in ['Low', 'Medium', 'High']:
        sub = merged[merged['demand_level'] == level]
        model = fit_rsm(sub)
        plot_response_surface(sub, model, suffix=f'_{level}')  # Save with level suffix

def main():
    df = load_and_prepare_data()

    # Compute centred versions of α and C
    df_cs = _center_scale(df, ['alpha', 'C'])

    # Base RSM
    base_formula = 'cost_diff ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C'
    rsm_model = smf.ols(base_formula, data=df).fit()
    print(rsm_model.summary())
    _save_latex_summary(rsm_model, 'rsm_base_summary.tex')
    cv_rmse = _cross_val_rmse(df, base_formula)
    print(f'5-fold CV RMSE (baseline RSM): {cv_rmse:.2f}')
    plot_response_surface(df, rsm_model)

    # Centred RSM for multicollinearity diagnostics
    centred_formula = 'cost_diff ~ alpha_c + C_c + I(alpha_c**2) + I(C_c**2) + alpha_c:C_c'
    centred_model = smf.ols(centred_formula, data=df_cs).fit()
    print(centred_model.summary())
    _save_latex_summary(centred_model, 'rsm_centred_summary.tex')

    # Demand characterisation
    char_df = pd.read_csv(CHAR_PATH)
    char_df = char_df.rename(columns={'num_customers': 'demand_customers', 'total_kg': 'demand_total_kg'})
    char_df['day_id'] = 'sales_' + char_df['day_id'] + '_demand'

    # Moderation & stratified analyses (unchanged)
    mod_model = demand_moderation(df, char_df)
    _save_latex_summary(mod_model, 'rsm_moderation_summary.tex')
    stratified_rsm(df, char_df)

if __name__ == "__main__":
    main()
