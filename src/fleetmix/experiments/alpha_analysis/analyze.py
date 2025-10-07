"""
Consolidated analysis script for alpha grid search results.
This script combines the functionality of several analysis scripts into a single,
configurable tool. It can be used to:
1.  Characterize the demand dataset.
2.  Perform full analysis including economic sweet spots, operational KPIs, etc.
3.  Generate enhanced heatmaps with RSM and win-probability overlays.
4.  Analyze demand robustness with scatter plots.
5.  Conduct Heterogeneous Treatment Effect (HTE) analysis.
Use the command-line interface to select which analysis to run.
Example:
    python experiments/alpha_analysis/analyze.py --run-all
    python experiments/alpha_analysis/analyze.py --characterize --economic-plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.spatial import ConvexHull, KDTree

from fleetmix.config import load_fleetmix_params

# =============================================================================
# --- Configuration ---
# =============================================================================

# Path to the raw JSON results files
RESULTS_RAW = Path("src/fleetmix/experiments/alpha_analysis/results/raw")

# Directory containing the raw daily demand CSV files
DATA_DIR = Path("src/fleetmix/benchmarking/datasets/case")

# Base output directory for all generated files
BASE_OUTPUT_DIR = Path("results/consolidated_analysis")

# Specific output directories for each analysis type
CHAR_OUTPUT_DIR = BASE_OUTPUT_DIR / "demand_characterization"
FULL_FIGS_DIR = BASE_OUTPUT_DIR / "full_analysis_figs"
TABLES_DIR = BASE_OUTPUT_DIR / "tables"
ENHANCED_HEATMAP_DIR = BASE_OUTPUT_DIR / "enhanced_heatmaps"
HTE_OUTPUT_DIR = BASE_OUTPUT_DIR / "hte_analysis"

# Create directories if they don't exist
for path in [
    BASE_OUTPUT_DIR,
    CHAR_OUTPUT_DIR,
    FULL_FIGS_DIR,
    TABLES_DIR,
    ENHANCED_HEATMAP_DIR,
    HTE_OUTPUT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Path to the default fleetmix configuration
BASE_CONFIG_PATH = Path("src/fleetmix/config/default_config_experiments.yaml.yaml")

# Depot location (from default_config_experiments.yaml.yaml)
DEPOT_LAT, DEPOT_LON = 4.7, -74.1
R_EARTH_KM = 6371.0

# Plotting style
sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 150})

# Optional: set random seed for reproducibility in CV
RNG = np.random.default_rng(42)

# =============================================================================
# --- Core Data Loading & Preparation ---
# =============================================================================


def load_results():
    """Load all existing JSON results with additional computations."""
    all_results = []
    if not RESULTS_RAW.exists():
        print(f"Error: Results directory not found at {RESULTS_RAW.resolve()}")
        return pd.DataFrame()

    json_files = list(RESULTS_RAW.glob("sales_*_demand_*.json"))
    print(f"Found {len(json_files)} result files in {RESULTS_RAW.resolve()}")

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                all_results.append(data)
        except json.JSONDecodeError:
            print(f"Skipping corrupted file: {json_path}")
            continue

    if not all_results:
        print("No results loaded.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} results")

    base_params = load_fleetmix_params(BASE_CONFIG_PATH)
    base_vehicle_spec = next(iter(base_params.problem.vehicles.values()))
    f_sc = float(base_vehicle_spec.fixed_cost)
    print(f"Using SCV fixed cost f_SC = {f_sc}")

    df["alpha_surcharge_pct"] = ((df["alpha"] - 1) * 100).round().astype(int)
    df["c_pct_scv"] = (100 * df["C"] / f_sc).round().astype(int)

    scv_df = df[df["fleet_type"] == "SCV"]
    if len(scv_df) > 0:
        scv_metrics = scv_df.agg(
            {
                "solver_cost": "mean",
                "cost_per_kg": "mean",
                "cost_per_drop": "mean",
                "total_vehicles": "mean",
                "total_fixed_cost": "mean",
                "total_variable_cost": "mean",
                "total_route_time_hours": "mean",
            }
        )
        for k, v in scv_metrics.items():
            if v > 0:
                df[f"{k}_index"] = df[k] / v

    def compute_total_capacity(row):
        vehicles_used = row["vehicles_used"]
        if row["fleet_type"] == "SCV":
            return sum(vehicles_used.values()) * 2700
        else:
            capacities = {"A": 2700, "B": 3300, "C": 4500}
            return sum(
                count * capacities.get(vt, 2700) for vt, count in vehicles_used.items()
            )

    df["total_capacity"] = df.apply(compute_total_capacity, axis=1)
    df["load_factor"] = df["total_demand_kg"] / df["total_capacity"].replace(0, np.nan)
    return df


def compute_stats(df, cost_col="solver_cost"):
    """Compute win rates and percentage savings."""
    mcv = df[df["fleet_type"] == "MCV"].copy()
    scv = (
        df[df["fleet_type"] == "SCV"][["day_id", cost_col]]
        .rename(columns={cost_col: f"{cost_col}_scv"})
        .set_index("day_id")
    )

    def group_stats(g):
        costs = g.merge(scv, left_on="day_id", right_index=True)
        if costs.empty:
            return pd.Series()
        diff = costs[f"{cost_col}_scv"] - costs[cost_col]
        rel_diff = diff / costs[f"{cost_col}_scv"]
        win_rate = (diff > 0).mean()
        avg_pct_wins = rel_diff[diff > 0].mean() * 100 if any(diff > 0) else 0
        return pd.Series(
            {
                "win_rate": win_rate,
                "avg_pct_savings": rel_diff.mean() * 100,
                "median_pct_savings_wins": rel_diff[diff > 0].median() * 100
                if any(diff > 0)
                else 0,
                "num_days": len(costs),
            }
        )

    return (
        mcv.groupby(["alpha_surcharge_pct", "c_pct_scv"])
        .apply(group_stats, include_groups=False)
        .reset_index()
    )


def get_savings_df(df, cost_col="solver_cost"):
    """Create a dataframe with daily percentage savings."""
    mcv = df[df["fleet_type"] == "MCV"].copy()
    scv = (
        df[df["fleet_type"] == "SCV"][["day_id", cost_col]]
        .rename(columns={cost_col: f"{cost_col}_scv"})
        .set_index("day_id")
    )
    merged = mcv.merge(scv, left_on="day_id", right_index=True)
    merged["pct_savings"] = (
        (merged[f"{cost_col}_scv"] - merged[cost_col]) / merged[f"{cost_col}_scv"] * 100
    )
    return merged


def load_and_prepare_data_for_hte():
    """Load and prepare data specifically for HTE analysis."""
    raw_df = load_results()
    scv = (
        raw_df[raw_df["fleet_type"] == "SCV"][["day_id", "solver_cost"]]
        .rename(columns={"solver_cost": "scv_cost"})
        .set_index("day_id")
    )
    mcv = raw_df[raw_df["fleet_type"] == "MCV"]
    df = mcv.merge(scv, left_on="day_id", right_index=True)
    df["cost_diff"] = df["scv_cost"] - df["solver_cost"]
    return df


# =============================================================================
# --- Analysis Sections ---
# =============================================================================


def run_dataset_characterization():
    """
    Run comprehensive dataset characterization, including summary statistics,
    histograms, scatter plots, spatial analysis, temporal analysis, and
    correlation heatmaps, generating several tables and figures.
    """
    print("=" * 60 + "\n1. RUNNING DATASET CHARACTERIZATION\n" + "=" * 60)

    daily_df = _aggregate_daily_stats()
    daily_df.to_csv(CHAR_OUTPUT_DIR / "daily_summary.csv", index=False)

    overview = {
        "num_days": len(daily_df),
        "customers": daily_df["num_customers"]
        .agg(["min", "max", "mean", "median"])
        .to_dict(),
        "total_kg": daily_df["total_kg"]
        .agg(["min", "max", "mean", "median"])
        .to_dict(),
    }
    print("Dataset Overview:")
    print(json.dumps(overview, indent=2))
    with open(CHAR_OUTPUT_DIR / "dataset_overview.json", "w") as fp:
        json.dump(overview, fp, indent=2)

    # --- Base Plots ---
    _char_histogram(
        daily_df["num_customers"],
        "Distribution of Daily Customer Count",
        "Number of Customers",
        "Number of Days",
        "hist_num_customers.png",
    )
    _char_histogram(
        daily_df["total_kg"],
        "Distribution of Daily Demand (kg)",
        "Total kg",
        "Number of Days",
        "hist_total_kg.png",
    )
    _scatter(
        daily_df["num_customers"],
        daily_df["total_kg"],
        "Customers vs Total Demand",
        "Number of Customers",
        "Total kg",
        "scatter_customers_vs_kg.png",
    )

    # --- Spatial Analysis ---
    spatial_records = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        df["distance_km"] = _haversine(df["Lat"], df["Lon"])
        df["day_id"] = day_id
        spatial_records.append(df[["day_id", "distance_km", "Lat", "Lon"]])

    all_spatial = pd.concat(spatial_records, ignore_index=True)
    _char_histogram(
        all_spatial["distance_km"],
        "Customer Distance to Depot",
        "Distance (km)",
        "Number of Customer Deliveries",
        "hist_distance_km.png",
    )

    plt.figure(figsize=(6, 6))
    sns.kdeplot(
        x=all_spatial["Lon"],
        y=all_spatial["Lat"],
        cmap="viridis",
        fill=True,
        bw_adjust=0.5,
        thresh=0.05,
    )
    plt.scatter([DEPOT_LON], [DEPOT_LAT], color="red", marker="*", label="Depot")
    plt.title("Spatial Density of Drops (KDE)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / "kde_spatial_density.png")
    plt.close()

    dist_stats = (
        all_spatial.groupby("day_id")["distance_km"]
        .agg(["mean", "median", "std", "max", "count"])
        .reset_index()
    )
    dist_stats.to_csv(CHAR_OUTPUT_DIR / "distance_stats_per_day.csv", index=False)

    # --- Temporal Analysis ---
    daily_df["weekday"] = pd.to_datetime(daily_df["date"]).dt.day_name()
    _create_daily_distribution_plot(
        daily_df, CHAR_OUTPUT_DIR / "fig_A5_weekday_distributions.png"
    )

    # --- Advanced Metrics and Figures ---
    geo_records = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df_day = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        coords = df_day[["Lon", "Lat"]].to_numpy()
        area_sq_km: float = 0.0
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                area_sq_km = float(hull.volume)  # area for 2D
            except Exception:
                pass
        geo_records.append(
            {
                "day_id": day_id,
                "geo_area_sq_km": area_sq_km,
                "demand_density": len(df_day) / area_sq_km if area_sq_km > 0 else 0,
            }
        )
    geo_df = pd.DataFrame(geo_records)
    corr_df = daily_df.merge(geo_df, on="day_id")
    dist_daily = dist_stats.rename(
        columns={
            "mean": "mean_distance_km",
            "std": "std_distance_km",
            "max": "max_distance_km",
        }
    )
    corr_df = corr_df.merge(dist_daily, on="day_id")

    dist_between_cust_records = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df_day = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        if len(df_day) > 1:
            coords_rad = np.radians(df_day[["Lat", "Lon"]].to_numpy())
            tree = KDTree(coords_rad)
            distances_result, _ = tree.query(coords_rad, k=2)
            # distances_result is always ndarray when coords_rad is 2D
            distances = np.asarray(distances_result)
            avg_dist_km = float(np.mean(distances[:, 1])) * R_EARTH_KM
        else:
            avg_dist_km = 0
        dist_between_cust_records.append(
            {"day_id": day_id, "avg_dist_between_customers_km": avg_dist_km}
        )
    dist_between_df = pd.DataFrame(dist_between_cust_records)
    corr_df = corr_df.merge(dist_between_df, on="day_id")

    # --- Correlation Heatmap & Appendix Figures ---
    _generate_appendix_figures(corr_df, daily_df)
    print(f"Characterization plots and summary saved to {CHAR_OUTPUT_DIR.resolve()}")


def run_economic_and_operational_plots(df):
    print("\n" + "=" * 60 + "\n2. RUNNING ECONOMIC & OPERATIONAL PLOTS\n" + "=" * 60)
    if df.empty:
        return

    plot_economic_sweet_spot(df, FULL_FIGS_DIR)
    plot_computational_footprint(df, FULL_FIGS_DIR)
    plot_urban_impact(df, FULL_FIGS_DIR)


def run_table_generation(df):
    print("\n" + "=" * 60 + "\n3. RUNNING TABLE GENERATION\n" + "=" * 60)
    if df.empty:
        return
    generate_story_telling_tables(df, TABLES_DIR)

    # Generate win rate savings maps for different cost metrics
    for cost_col, metric_name in [
        ("solver_cost", "total_cost"),
    ]:
        plot_win_rate_savings_map(df, TABLES_DIR, cost_col, metric_name)


def run_advanced_plots(df_full, df_hte):
    print("\n" + "=" * 60 + "\n4. RUNNING ADVANCED & ENHANCED PLOTS\n" + "=" * 60)
    if df_full.empty:
        return
    stats_df = compute_stats(df_full)
    plot_demand_stratified_heatmaps(df_full, ENHANCED_HEATMAP_DIR)
    plot_safe_zones_heatmap(df_full, stats_df, ENHANCED_HEATMAP_DIR)


def run_hte_analysis(df_hte):
    print("\n" + "=" * 60 + "\n5. RUNNING HTE ANALYSIS\n" + "=" * 60)
    if df_hte.empty:
        return
    rsm_model = fit_rsm(df_hte)
    print("\n--- Base RSM Model Summary ---\n", rsm_model.summary())


# --- Plotting Functions ---


def plot_economic_sweet_spot(df, figs_dir):
    stats = compute_stats(df)
    if stats.empty:
        return
    pivot_pct = stats.pivot(
        index="alpha_surcharge_pct", columns="c_pct_scv", values="avg_pct_savings"
    ).fillna(0)
    pivot_wr = stats.pivot(
        index="alpha_surcharge_pct", columns="c_pct_scv", values="win_rate"
    )

    # Build annotation with win days: "12%\n23/30"
    days = (
        int(stats["num_days"].max()) if "num_days" in stats.columns else 70
    )  # fallback
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
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Average % Cost Savings"},
    )
    X, Y = np.meshgrid(pivot_pct.columns, pivot_pct.index)
    if np.any(pivot_pct.values):
        plt.contour(
            X, Y, pivot_pct.values, levels=[0], colors="black", linestyles="solid"
        )
    plt.title("Economic Sweet-Spot: Avg %-Savings (colour) | Wins/Days (text)")
    plt.xlabel("Setup Cost C (% of SCV cap-ex)")
    plt.ylabel("Vehicle Surcharge α (%)")
    plt.tight_layout()
    plt.savefig(figs_dir / "economic_sweet_spot.png")
    plt.close()


def plot_computational_footprint(df, figs_dir):
    """Plot solver runtime vs. instance size."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="num_customers",
        y="solver_runtime_sec",
        hue="fleet_type",
        style="fleet_type",
        s=100,
        alpha=0.7,
    )
    plt.yscale("log")
    plt.title("Solve Time vs Instance Size")
    plt.xlabel("Number of Customers")
    plt.ylabel("Solve Time (seconds, log scale)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "computational_footprint.png")
    plt.close()


def plot_urban_impact(df, figs_dir):
    """Shows the societal benefit of fewer vehicles on roads."""
    interesting_configs = [
        ("SCV", 0, 0),
        ("MCV", 0, 10),
        ("MCV", 25, 10),
        ("MCV", 50, 20),
    ]
    scv_baseline_vehicles = df[df.fleet_type == "SCV"]["total_vehicles"].mean()
    plot_data = []
    for fleet_type, alpha, c in interesting_configs:
        if fleet_type == "SCV":
            subset = df[df.fleet_type == "SCV"]
            label = "SCV Baseline"
        else:
            subset = df[
                (df.fleet_type == "MCV")
                & (df.alpha_surcharge_pct == alpha)
                & (df.c_pct_scv == c)
            ]
            label = f"MCV (α={alpha}%, C={c}%)"
        if not subset.empty:
            avg_vehicles = subset.total_vehicles.mean()
            reduction = (
                (1 - avg_vehicles / scv_baseline_vehicles) * 100
                if scv_baseline_vehicles
                else 0
            )
            plot_data.append(
                {
                    "label": label,
                    "avg_vehicles": avg_vehicles,
                    "reduction_pct": reduction,
                }
            )
    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["label"], plot_df["avg_vehicles"], color="lightblue")
    for bar, reduction in zip(bars[1:], plot_df["reduction_pct"][1:]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"-{reduction:.0f}%",
            ha="center",
            va="bottom",
            color="green",
            weight="bold",
        )
    plt.axhline(
        scv_baseline_vehicles,
        color="red",
        linestyle="--",
        label=f"SCV Mean: {scv_baseline_vehicles:.1f}",
    )
    plt.ylabel("Average Number of Vehicles")
    plt.title("Urban Impact: Fleet Size Reduction")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "urban_impact.png")
    plt.close()


def generate_story_telling_tables(df, tables_dir):
    stats = compute_stats(df)
    # Table A1
    break_evens = []
    for c in sorted(stats["c_pct_scv"].unique()):
        c_data = stats[stats["c_pct_scv"] == c]
        eligible = c_data[c_data["win_rate"] >= 0.5]
        max_alpha = (
            eligible["alpha_surcharge_pct"].max() if not eligible.empty else np.nan
        )
        break_evens.append({"C (% of SCV)": c, "Break-even α (%)": max_alpha})
    pd.DataFrame(break_evens).to_markdown(
        tables_dir / "table_a1_thresholds.md", index=False
    )


def plot_win_rate_savings_map(
    df, figs_dir, cost_col="solver_cost", metric_name="total_cost"
):
    """Generate the win rate and savings map showing win days (e.g., 30/70)."""
    print(f"  - Generating win rate savings map for {metric_name}")
    stats = compute_stats(df, cost_col)
    if stats.empty:
        return

    # Pivot tables for colour (average %-savings) and win-rate
    pivot_pct = stats.pivot(
        index="alpha_surcharge_pct", columns="c_pct_scv", values="avg_pct_savings"
    ).round(1)
    pivot_wr = stats.pivot(
        index="alpha_surcharge_pct", columns="c_pct_scv", values="win_rate"
    )

    # Build annotation: "12%\n23/30"
    days = int(stats["num_days"].max()) if not stats.empty else 70
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
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Average %-Savings vs SCV"},
    )
    plt.title(
        f"MCV Advantage ({metric_name}): Avg %-Savings (colour) | Wins/Days (text)"
    )
    plt.ylabel("MCV Surcharge % (from α)")
    plt.xlabel("C as % of SCV fixed cost")
    plt.tight_layout()
    plt.savefig(figs_dir / f"win_rate_savings_map_{metric_name}.png")
    plt.close()


def fit_rsm(df: pd.DataFrame, robust: bool = True):
    """Fit a second-order Response-Surface model on coded factors.

    The continuous factors alpha (vehicle surcharge) and C (setup cost) are
    mean-centered and scaled to half-range units (coded to approximately
    the [-1, +1] interval).  This is the standard practice in RSM and
    drastically reduces multicollinearity between linear, quadratic and
    interaction terms, lowering the design-matrix condition number.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *alpha*, *C* and *cost_diff* columns.
    robust : bool, default True
        If ``True`` the model is fitted with heteroscedasticity-consistent
        (HC3) standard errors, a simple safeguard when the equal-variance
        assumption is doubtful.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted model with an extra ``code_info`` attribute storing the
        (centre, half-range) tuples for each factor so that downstream
        prediction utilities can transform new points consistently.
    """

    data = df.copy()

    # ------------------------------------------------------------------
    # Code/scale the factors to mitigate numerical issues
    # ------------------------------------------------------------------
    code_info: dict[str, tuple[float, float]] = {}
    for var in ["alpha", "C"]:
        centre = data[var].mean()
        half_range = (data[var].max() - data[var].min()) / 2.0 or 1.0
        data[var] = (data[var] - centre) / half_range
        code_info[var] = (centre, half_range)

    # Fit the canonical second-order RSM
    model = smf.ols(
        "cost_diff ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C",
        data=data,
    )

    results = model.fit(cov_type="HC3") if robust else model.fit()

    # Attach coding information for use during prediction
    results.code_info = code_info
    return results


def fit_win_probability_model(df):
    """Fit logistic regression for P(win)."""
    df["win"] = (df["cost_diff"] > 0).astype(int)
    return smf.logit("win ~ alpha + C + I(alpha**2) + I(C**2) + alpha:C", data=df).fit(
        disp=0
    )


def plot_demand_stratified_heatmaps(df_full, output_dir):
    """Small multiples by demand strata."""
    print("  - Generating demand-stratified heatmaps")
    char_df = pd.read_csv(CHAR_OUTPUT_DIR / "daily_summary.csv")
    char_df["day_id"] = "sales_" + char_df["day_id"] + "_demand"
    merged = df_full.merge(char_df[["day_id", "total_kg"]], on="day_id")
    merged["demand_tercile"] = pd.qcut(
        merged["total_kg"], 3, labels=["Low", "Medium", "High"]
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))
    for i, (demand_level, ax) in enumerate(zip(["Low", "Medium", "High"], axes)):
        level_days = merged[merged["demand_tercile"] == demand_level]["day_id"].unique()
        level_data = df_full[df_full["day_id"].isin(level_days)]
        level_stats = compute_stats(level_data)
        if level_stats.empty:
            continue
        pivot_pct = level_stats.pivot(
            index="alpha_surcharge_pct", columns="c_pct_scv", values="avg_pct_savings"
        )
        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Avg %-Savings"},
            ax=ax,
            vmin=-30,
            vmax=30,
        )
        ax.set_title(
            f"{demand_level} Demand Days (avg: {merged[merged['demand_tercile'] == demand_level]['total_kg'].mean():.0f} kg)"
        )
        ax.set_xlabel("C as % of SCV fixed cost" if i == 2 else "")
    plt.suptitle("MCV Advantage Across Demand Levels", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_demand_multiples.png")
    plt.close()


def plot_safe_zones_heatmap(df_full, stats_df, output_dir):
    """Convey managerial 'safe zones'."""
    print("  - Generating safe zones heatmap")
    WIN_PROB_THRESHOLD, MIN_SAVINGS_THRESHOLD = 0.9, 5.0
    savings_df = get_savings_df(df_full)
    safe_zone_data = []
    for (alpha_pct, c_pct), group in savings_df.groupby(
        ["alpha_surcharge_pct", "c_pct_scv"]
    ):
        safe_zone_data.append(
            {
                "alpha_surcharge_pct": alpha_pct,
                "c_pct_scv": c_pct,
                "is_safe": ((group["pct_savings"] > 0).mean() >= WIN_PROB_THRESHOLD)
                and (group["pct_savings"].mean() >= MIN_SAVINGS_THRESHOLD),
            }
        )
    safe_df = pd.DataFrame(safe_zone_data)
    pivot_pct = stats_df.pivot(
        index="alpha_surcharge_pct", columns="c_pct_scv", values="avg_pct_savings"
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Average %-Savings vs SCV"},
        ax=ax,
    )
    for _, row in safe_df[safe_df["is_safe"]].iterrows():
        try:
            y_idx, x_idx = (
                list(pivot_pct.index).index(row["alpha_surcharge_pct"]),
                list(pivot_pct.columns).index(row["c_pct_scv"]),
            )
            ax.add_patch(
                patches.Rectangle(
                    (x_idx, y_idx),
                    1,
                    1,
                    linewidth=3,
                    edgecolor="darkgreen",
                    facecolor="none",
                )
            )
        except ValueError:
            continue
    ax.legend(
        handles=[
            patches.Patch(
                color="darkgreen",
                label=f"Safe Zone: P(win)≥{WIN_PROB_THRESHOLD:.0%} & Avg Savings≥{MIN_SAVINGS_THRESHOLD}%",
            )
        ],
        loc="upper right",
    )
    ax.set_title("MCV Advantage with Managerial Safe Zones")
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_safe_zones.png")
    plt.close()


def _haversine(lat1, lon1):
    """Vectorised haversine distance in km to depot."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = math.radians(DEPOT_LAT), math.radians(DEPOT_LON)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def _scatter(
    x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: str
):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x, y=y)
    sns.regplot(x=x, y=y, scatter=False, color="red", ci=None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / filename)
    plt.close()


def _create_daily_distribution_plot(daily_df: pd.DataFrame, save_path: Path):
    """Create a side-by-side boxplot of customers and demand by weekday."""
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(
        x="weekday",
        y="num_customers",
        data=daily_df,
        order=weekday_order,
        ax=axes[0],
        hue="weekday",
        palette="viridis",
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        x="weekday",
        y="num_customers",
        data=daily_df,
        order=weekday_order,
        ax=axes[0],
        color=".25",
        size=5,
        alpha=0.7,
    )
    axes[0].set_title("Distribution of Daily Customers by Weekday")
    axes[1].tick_params(axis="x", rotation=45)
    sns.boxplot(
        x="weekday",
        y="total_kg",
        data=daily_df,
        order=weekday_order,
        ax=axes[1],
        hue="weekday",
        palette="viridis",
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        x="weekday",
        y="total_kg",
        data=daily_df,
        order=weekday_order,
        ax=axes[1],
        color=".25",
        size=5,
        alpha=0.7,
    )
    axes[1].set_title("Distribution of Daily Demand (kg) by Weekday")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Daily Customer and Demand Distribution by Weekday", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _generate_appendix_figures(corr_df, daily_df):
    """Generate all appendix figures related to dataset characterization."""
    # Fig A1
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(
        corr_df["date"], corr_df["total_kg"], marker="o", linestyle="-", markersize=4
    )
    axes[0].set_title("Panel A: Daily Total Demand Volume")
    sns.boxplot(
        x="weekday",
        y="total_kg",
        data=corr_df,
        order=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        ax=axes[1],
        hue="weekday",
        legend=False,
    )
    axes[1].set_title("Panel B: Day-of-Week Effects on Total Demand")
    axes[2].plot(
        corr_df["date"],
        corr_df["num_customers"],
        marker="o",
        linestyle="-",
        color="teal",
    )
    axes[2].set_title("Panel C: Daily Customer Count")
    fig.suptitle("Figure A1: Temporal Demand Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / "fig_A1_temporal_patterns.png")
    plt.close()

    # Fig A2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(daily_df["num_customers"], kde=True, ax=axes[0])
    axes[0].set_title("Panel A: Daily Customer Counts")
    sns.histplot(daily_df["total_kg"], kde=True, ax=axes[1])
    axes[1].set_title("Panel B: Daily Total Volume")
    sns.histplot(corr_df["geo_area_sq_km"], kde=True, ax=axes[2])
    axes[2].set_title("Panel C: Geographic Coverage per Day")
    fig.suptitle("Figure A2: Demand Scale and Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / "fig_A2_demand_scale.png")
    plt.close()

    # Fig A3
    corr_df["cv_total_kg"] = corr_df["std_kg"] / corr_df["mean_kg"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(corr_df["date"], corr_df["cv_total_kg"], marker="o", linestyle="-")
    axes[0].set_title("Panel A: Coefficient of Variation (Order Size)")
    sns.scatterplot(data=corr_df, x="geo_area_sq_km", y="demand_density", ax=axes[1])
    axes[1].set_title("Panel B: Demand Density")
    fig.suptitle("Figure A3: Demand Variability and Complexity", fontsize=16)
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / "fig_A3_demand_variability.png")
    plt.close()


def _aggregate_daily_stats() -> pd.DataFrame:
    """Parse all CSV files and return per-day statistics as a DataFrame."""
    records = []
    csv_files = sorted(DATA_DIR.glob("*_demand.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No demand CSV files found in {DATA_DIR}")
    for path in csv_files:
        try:
            records.append(_summarise_day(path))
        except Exception as err:
            print(f"Skipping {path.name}: {err}")
    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.reset_index(drop=True)


def _char_histogram(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    logx: bool = False,
):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True, bins=30)
    if logx:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(CHAR_OUTPUT_DIR / filename)
    plt.close()


def _summarise_day(csv_path: Path) -> dict:
    """Return basic statistics for a single demand CSV file."""
    df = pd.read_csv(csv_path)
    day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
    return {
        "day_id": day_id,
        "date": pd.to_datetime(day_id, errors="coerce"),
        "num_customers": len(df),
        "total_kg": df["Kg"].sum(),
        "mean_kg": df["Kg"].mean(),
        "median_kg": df["Kg"].median(),
        "std_kg": df["Kg"].std(),
        "min_kg": df["Kg"].min(),
        "p10_kg": np.percentile(df["Kg"], 10),
        "p90_kg": np.percentile(df["Kg"], 90),
        "max_kg": df["Kg"].max(),
    }


# === NEW ANALYSIS HELPER =====================================================


def demand_interaction_test(df_full: pd.DataFrame):
    """Test whether %-savings vary with demand level (terciles).

    This routine first derives the percentage savings vs. SCV for every
    α–C–day observation (using :pyfunc:`get_savings_df`).  It then merges the
    total daily demand, assigns each day to a *Low/Medium/High* tercile and
    fits the interaction model requested in the specification.

    Parameters
    ----------
    df_full : pd.DataFrame
        Raw results dataframe containing all α–C–day runs (including SCV).

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model with HC3 robust standard errors.
    """
    # ------------------------------------------------------------------
    # Compute %-savings relative to SCV baseline (MCV rows only)
    # ------------------------------------------------------------------
    savings_df = get_savings_df(df_full)  # adds `pct_savings` column

    # ------------------------------------------------------------------
    # Attach demand information and create terciles
    # ------------------------------------------------------------------
    char_df = pd.read_csv(
        CHAR_OUTPUT_DIR / "daily_summary.csv", usecols=["day_id", "total_kg"]
    )
    char_df["day_id"] = "sales_" + char_df["day_id"] + "_demand"

    merged = savings_df.merge(char_df, on="day_id", how="left")
    merged = merged.dropna(subset=["total_kg"])  # ensure demand kg is present

    merged["demand_tercile"] = pd.qcut(
        merged["total_kg"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )

    # ------------------------------------------------------------------
    # Fit OLS with HC3 robust covariance
    # ------------------------------------------------------------------
    formula = (
        "pct_savings ~ alpha_surcharge_pct + c_pct_scv + demand_tercile + "
        "alpha_surcharge_pct:demand_tercile + c_pct_scv:demand_tercile"
    )
    model = smf.ols(formula, data=merged).fit(cov_type="HC3")

    # ------------------------------------------------------------------
    # Print diagnostics and joint F-test for interaction terms
    # ------------------------------------------------------------------
    print("\n" + "=" * 60 + "\n6. DEMAND INTERACTION TEST\n" + "=" * 60)
    print(model.summary())

    # ------------------------------------------------------------------
    # Robust joint F-test for *all* interaction terms (Low & High)
    # ------------------------------------------------------------------
    param_names = list(model.params.index)
    interaction_terms = [
        "alpha_surcharge_pct:demand_tercile[T.Low]",
        "alpha_surcharge_pct:demand_tercile[T.High]",
        "c_pct_scv:demand_tercile[T.Low]",
        "c_pct_scv:demand_tercile[T.High]",
    ]

    # Keep only those actually present in the fitted model
    available_terms = [t for t in interaction_terms if t in param_names]
    R = np.zeros((len(available_terms), len(param_names)))
    for i, term in enumerate(available_terms):
        R[i, param_names.index(term)] = 1.0

    ftest = model.f_test(R)

    print("\n--- Joint F-test for interaction terms ---\n", ftest)

    p_val = float(ftest.pvalue)
    if p_val > 0.05:
        print("✅  No interaction: surface is demand-robust (p = {:.3f})".format(p_val))
    else:
        print("⚠️  Interaction detected (p = {:.3f})".format(p_val))

    # ------------------------------------------------------------------
    # Effect-size lens: quantify the *largest* possible shift within data
    # ------------------------------------------------------------------
    beta = model.params.get("c_pct_scv:demand_tercile[T.High]", 0.0)
    max_C = merged["c_pct_scv"].max()
    max_shift_pp = abs(beta * max_C)
    tolerance_pp = 2.0  # business-defined negligible band

    verdict = (
        "✅  practically equivalent"
        if max_shift_pp < tolerance_pp
        else "⚠️  exceeds tolerance"
    )
    print(
        f"\nEffect-size lens ➜ max shift = {max_shift_pp:.2f} pp (tolerance ±{tolerance_pp:.2f} pp)  →  {verdict}"
    )

    return model, max_shift_pp


# =============================================================================
# --- Main CLI ---
# =============================================================================


def main(config_path: Path | None = None):
    parser = argparse.ArgumentParser(
        description="Consolidated analysis for fleet-mix experiments (runs everything by default)."
    )
    # Keeping this flag for forward-compatibility / explicitness, but it is optional.
    parser.add_argument(
        "--run-all",
        action="store_true",
        default=True,
        help="Run all analyses (default behaviour). This is currently the only available option.",
    )
    _ = parser.parse_args()  # parsed for completeness, value is always True for now

    # --- Execute full analysis pipeline ---
    run_dataset_characterization()

    df = load_results()
    hte_df = load_and_prepare_data_for_hte()

    run_economic_and_operational_plots(df)
    run_table_generation(df)
    run_advanced_plots(df, hte_df)
    demand_interaction_test(df)
    run_hte_analysis(hte_df)

    print("\nAnalysis script finished.")


if __name__ == "__main__":
    main()
