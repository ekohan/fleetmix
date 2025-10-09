"""
Fleet Composition Analysis for Mixed Fleet Optimization (Experiment 2).

This script analyzes how fleet composition (MCV vs SCV) varies across the (alpha, C)
parameter space, complementing the cost-focused analysis from Experiment 1.

Focus: Fleet composition dynamics, MCV adoption patterns, and technology dominance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# Paths
PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PKG_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "summary_mixed.parquet"
OUTPUT_DIR = RESULTS_DIR / "fleet_composition_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Get project root (4 levels up: alpha_analysis -> experiments -> fleetmix -> src -> root)
PROJECT_ROOT = PKG_DIR.parent.parent.parent.parent
DEMAND_DIR = PROJECT_ROOT / "src" / "fleetmix" / "benchmarking" / "datasets" / "case"


def compute_multi_good_proportion(instance_name: str) -> float:
    """
    Compute proportion of customers requiring multiple temperature classes.

    Args:
        instance_name: e.g., 'avg_daily_demand__2023_09_01'

    Returns:
        Proportion of multi-good customers (0.0-1.0)
    """
    try:
        # Build path to demand CSV
        demand_path = DEMAND_DIR / f"{instance_name}.csv"

        if not demand_path.exists():
            return 0.5  # Default fallback if file not found

        # Load customer demand data
        customers_df = pd.read_csv(demand_path)

        # Check if this is transaction format (ClientID, ProductType) or demand format (*_Demand columns)
        if "ClientID" in customers_df.columns and "ProductType" in customers_df.columns:
            # Transaction format: each row is a delivery, ProductType is the temperature class
            # Group by customer and count unique product types
            products_per_customer = customers_df.groupby("ClientID")[
                "ProductType"
            ].nunique()

            # Calculate proportion with >= 2 product types
            multi_good_customers = (products_per_customer >= 2).sum()
            total_customers = len(products_per_customer)

            return (
                multi_good_customers / total_customers if total_customers > 0 else 0.0
            )
        else:
            # Demand format: columns like 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand'
            demand_cols = [c for c in customers_df.columns if c.endswith("_Demand")]

            if not demand_cols:
                return 0.5  # No demand columns found

            # Count temperature classes per customer (demand > 0)
            num_goods_per_customer = (customers_df[demand_cols] > 0).sum(axis=1)

            # Calculate proportion with >= 2 temperature classes
            multi_good_customers = (num_goods_per_customer >= 2).sum()
            total_customers = len(customers_df)

            return (
                multi_good_customers / total_customers if total_customers > 0 else 0.0
            )

    except Exception as e:
        print(f"Error computing multi-good proportion for {instance_name}: {e}")
        return 0.5  # Fallback value


def load_mixed_results():
    """Load and prepare mixed fleet data."""
    df = pd.read_parquet(SUMMARY_PATH)

    # Separate baselines and mixed results
    scv_baseline = df[df["fleet_type"] == "SCV_BASE"].copy()
    mixed = df[df["fleet_type"] == "MIXED"].copy()

    # Compute multi-good proportion for each instance
    print("Computing multi-good customer proportions...")
    instance_multi_good = {}
    for instance in mixed["instance"].unique():
        instance_multi_good[instance] = compute_multi_good_proportion(instance)

    # Add to dataframes
    mixed["multi_good_pct"] = mixed["instance"].map(instance_multi_good)
    scv_baseline["multi_good_pct"] = scv_baseline["instance"].map(instance_multi_good)

    print(f"Loaded {len(mixed)} mixed fleet observations")
    print(f"  - {mixed['instance'].nunique()} unique demand days")
    print(f"  - {mixed['alpha'].nunique()} alpha values")
    print(f"  - {mixed['C'].nunique()} C values")
    print(
        f"  - Multi-good % range: {mixed['multi_good_pct'].min():.1%} - {mixed['multi_good_pct'].max():.1%}"
    )

    return mixed, scv_baseline


def compute_composition_metrics(df):
    """Compute aggregated metrics by (alpha, C)."""

    def classify_regime(mcv_share):
        """Classify fleet composition regime."""
        if mcv_share >= 0.99:
            return "Pure MCV"
        elif mcv_share >= 0.50:
            return "MCV-dominant"
        elif mcv_share >= 0.10:
            return "True Mixed"
        else:
            return "SCV-dominant"

    # Add regime classification
    df = df.copy()
    df["regime"] = df["mcv_share"].apply(classify_regime)

    # Aggregate by (alpha, C)
    metrics = (
        df.groupby(["alpha", "C"])
        .agg(
            {
                "mcv_share": ["mean", "median", "std", "min", "max"],
                "mcv_vehicles": [
                    (
                        "adoption_days",
                        lambda x: (x > 0).sum(),
                    ),  # Days with at least 1 MCV
                    (
                        "pure_mcv_days",
                        lambda x: (df.loc[x.index, "mcv_share"] >= 0.99).sum(),
                    ),
                    ("mean", "mean"),
                ],
                "scv_vehicles": "mean",
                "total_vehicles": ["mean", "median", "std"],
                "total_cost": "mean",
                "delta_cost_pct_vs_scv": ["mean", "median"],
                "instance": "count",  # Number of observations (should be 70)
            }
        )
        .reset_index()
    )

    # Flatten column names
    metrics.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in metrics.columns.values
    ]

    # Compute adoption probability (fraction of days with MCV > 0)
    metrics["adoption_probability"] = (
        metrics["mcv_vehicles_adoption_days"] / metrics["instance_count"]
    )

    # Compute regime distribution
    regime_dist = df.groupby(["alpha", "C", "regime"]).size().unstack(fill_value=0)
    regime_dist = (
        regime_dist.div(regime_dist.sum(axis=1), axis=0) * 100
    )  # Convert to percentages

    print("\n=== Composition Metrics Summary ===")
    print(f"Total parameter combinations: {len(metrics)}")
    print("\nMCV Share statistics:")
    print(f"  Mean: {metrics['mcv_share_mean'].mean():.1%}")
    print(
        f"  Range: {metrics['mcv_share_mean'].min():.1%} - {metrics['mcv_share_mean'].max():.1%}"
    )
    print("\nAdoption days:")
    print(f"  Mean: {metrics['mcv_vehicles_adoption_days'].mean():.1f}/70")
    print(
        f"  Range: {metrics['mcv_vehicles_adoption_days'].min()}-{metrics['mcv_vehicles_adoption_days'].max()}/70"
    )

    return metrics, regime_dist, df


def chart1_mcv_adoption_landscape(metrics):
    """
    Chart 1: Dual-layer heatmap showing MCV adoption prevalence and composition.
    Background: Average MCV share
    Annotations: MCV share % and adoption days count
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Prepare pivot tables
    pivot_share = metrics.pivot(index="alpha", columns="C", values="mcv_share_mean")
    pivot_adoption = metrics.pivot(
        index="alpha", columns="C", values="mcv_vehicles_adoption_days"
    )
    pivot_pure_mcv = metrics.pivot(
        index="alpha", columns="C", values="mcv_vehicles_pure_mcv_days"
    )

    # Create heatmap background (MCV share)
    im = ax.imshow(
        pivot_share.values * 100,
        cmap="YlGnBu",
        aspect="auto",
        vmin=0,
        vmax=100,
        origin="upper",
    )

    # Add annotations
    for i in range(len(pivot_share.index)):
        for j in range(len(pivot_share.columns)):
            share_pct = pivot_share.iloc[i, j] * 100
            adoption_days = int(pivot_adoption.iloc[i, j])
            pure_days = int(pivot_pure_mcv.iloc[i, j])

            # Determine text color for readability
            text_color = "white" if share_pct > 50 else "black"

            # Multi-line annotation
            text = f"{share_pct:.0f}%\n{adoption_days}/70\n({pure_days}★)"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
            )

    # Formatting
    ax.set_xticks(np.arange(len(pivot_share.columns)))
    ax.set_yticks(np.arange(len(pivot_share.index)))
    ax.set_xticklabels([f"{int(c)}%" for c in pivot_share.columns])
    ax.set_yticklabels(
        [f"{a:.1f}" if a != int(a) else f"{int(a)}.0" for a in pivot_share.index]
    )

    ax.set_xlabel(
        "C: Compartment Setup Cost (% of SCV fixed cost)", fontsize=12, weight="bold"
    )
    ax.set_ylabel("α: MCV Fixed Cost Multiplier", fontsize=12, weight="bold")
    ax.set_title(
        "MCV Adoption Landscape\nValues: MCV Share% | Adoption Days/70 | (Pure MCV Days★)",
        fontsize=14,
        weight="bold",
        pad=20,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average MCV Share (%)", fontsize=11)

    # Grid
    ax.set_xticks(np.arange(len(pivot_share.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_share.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "chart1_mcv_adoption_landscape.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Chart 1 saved: {output_path}")

    return fig


def chart2_fleet_composition_regimes(regime_dist, metrics):
    """
    Chart 2: Stacked bar chart showing distribution of fleet composition regimes.
    """
    # Select representative parameter points across the grid
    selected_params = [
        (1.0, 0),
        (1.0, 20),
        (1.1, 0),
        (1.1, 30),
        (1.2, 10),
        (1.2, 40),
        (1.3, 20),
        (1.3, 50),
        (1.4, 30),
        (1.5, 40),
        (1.6, 50),
        (1.8, 50),
        (2.0, 50),
    ]

    # Filter data
    regime_subset = regime_dist.loc[selected_params].reset_index()

    # Create labels
    labels = [f"α={a:.1f}\nC={int(c)}%" for a, c in selected_params]

    # Prepare data for stacking
    regimes = ["Pure MCV", "MCV-dominant", "True Mixed", "SCV-dominant"]
    regime_colors = {
        "Pure MCV": "#27ae60",  # Dark green
        "MCV-dominant": "#7dcea0",  # Light green
        "True Mixed": "#f39c12",  # Orange
        "SCV-dominant": "#e74c3c",  # Red
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Stacked bars
    bottom = np.zeros(len(selected_params))
    for regime in regimes:
        if regime in regime_subset.columns:
            values = regime_subset[regime].values
            ax.bar(
                labels,
                values,
                bottom=bottom,
                label=regime,
                color=regime_colors[regime],
                edgecolor="white",
                linewidth=1.5,
            )

            # Add percentage labels in the middle of each segment
            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 3:  # Only label if segment is large enough
                    ax.text(
                        i,
                        bot + val / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        weight="bold",
                    )

            bottom += values

    ax.set_ylabel("Distribution of Fleet Regimes (%)", fontsize=12, weight="bold")
    ax.set_xlabel(
        "Parameter Configuration (α: MCV Cost Multiplier, C: Compartment Cost)",
        fontsize=12,
        weight="bold",
    )
    ax.set_title(
        "Fleet Composition Regime Classification Across Parameter Space\n"
        + "70 demand days per configuration",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_ylim(0, 100)

    # Rotate x-labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "chart2_fleet_composition_regimes.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 2 saved: {output_path}")

    return fig


def chart3_mcv_adoption_probability_surface(metrics):
    """
    Chart 3: Contour plot showing MCV adoption probability as function of (α, C).
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Prepare data
    alpha_vals = metrics["alpha"].values
    c_vals = metrics["C"].values
    adoption_prob = metrics["adoption_probability"].values

    # Create grid for interpolation
    alpha_grid = np.linspace(alpha_vals.min(), alpha_vals.max(), 100)
    c_grid = np.linspace(c_vals.min(), c_vals.max(), 100)
    alpha_mesh, c_mesh = np.meshgrid(alpha_grid, c_grid)

    # Interpolate
    prob_mesh = griddata(
        (alpha_vals, c_vals), adoption_prob, (alpha_mesh, c_mesh), method="cubic"
    )

    # Create filled contour plot
    levels = np.linspace(0, 1, 21)
    contourf = ax.contourf(
        alpha_mesh, c_mesh, prob_mesh, levels=levels, cmap="RdYlGn", alpha=0.8
    )

    # Add contour lines at key adoption thresholds
    contour_lines = ax.contour(
        alpha_mesh,
        c_mesh,
        prob_mesh,
        levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        colors="black",
        linewidths=2,
        linestyles="--",
    )
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%0.0f%%", inline_spacing=10)

    # Highlight 50% adoption frontier (break-even)
    contour_50 = ax.contour(
        alpha_mesh, c_mesh, prob_mesh, levels=[0.5], colors="red", linewidths=3
    )

    # Scatter actual data points
    ax.scatter(
        alpha_vals,
        c_vals,
        c="black",
        s=50,
        marker="o",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.7,
    )

    # Formatting
    ax.set_xlabel("α: MCV Fixed Cost Multiplier", fontsize=12, weight="bold")
    ax.set_ylabel("C: Compartment Setup Cost (% of SCV)", fontsize=12, weight="bold")
    ax.set_title(
        "MCV Adoption Probability Surface\n"
        + "Probability = (Days with MCV > 0) / 70 days | Red line = 50% adoption frontier",
        fontsize=14,
        weight="bold",
        pad=20,
    )

    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("Adoption Probability", fontsize=11)
    cbar.set_ticks(np.linspace(0, 1, 11).tolist())
    cbar.set_ticklabels([f"{int(p * 100)}%" for p in np.linspace(0, 1, 11)])

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "chart3_mcv_adoption_probability.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 3 saved: {output_path}")

    return fig


def chart4_fleet_composition_variability(df_full):
    """
    Chart 4: Small multiples showing MCV share variation across demand days.
    """
    # Select representative parameter combinations
    selected_params = [
        (1.0, 0),
        (1.1, 0),
        (1.2, 0),
        (1.3, 0),
        (1.0, 20),
        (1.1, 20),
        (1.2, 20),
        (1.3, 20),
        (1.5, 30),
        (1.6, 40),
        (1.8, 50),
        (2.0, 50),
    ]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (alpha, C) in enumerate(selected_params):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        # Filter data for this parameter combination
        subset = df_full[(df_full["alpha"] == alpha) & (df_full["C"] == C)].copy()

        if len(subset) == 0:
            continue

        # Sort by instance (demand day)
        subset = subset.sort_values("instance").reset_index(drop=True)

        # Extract date from instance for x-axis
        subset["day_index"] = range(len(subset))

        # Color by regime
        regime_colors = {
            "Pure MCV": "#27ae60",
            "MCV-dominant": "#7dcea0",
            "True Mixed": "#f39c12",
            "SCV-dominant": "#e74c3c",
        }
        colors = [regime_colors[r] for r in subset["regime"]]

        # Plot
        ax.scatter(
            subset["day_index"],
            subset["mcv_share"] * 100,
            c=colors,
            s=30,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add mean line
        mean_share = subset["mcv_share"].mean() * 100
        ax.axhline(
            mean_share,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_share:.0f}%",
        )

        # Formatting
        ax.set_xlim(-1, len(subset))
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Demand Day Index", fontsize=9)
        ax.set_ylabel("MCV Share (%)", fontsize=9)
        ax.set_title(f"α={alpha:.1f}, C={int(C)}%", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Fleet Composition Variability Across Demand Days\n"
        + "Each point = one demand day, colored by composition regime",
        fontsize=14,
        weight="bold",
        y=0.995,
    )

    # Add legend for regime colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="black", label="Pure MCV (≥99% MCV)"),
        Patch(
            facecolor="#7dcea0", edgecolor="black", label="MCV-dominant (50-99% MCV)"
        ),
        Patch(facecolor="#f39c12", edgecolor="black", label="True Mixed (10-50% MCV)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="SCV-dominant (<10% MCV)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Save
    output_path = OUTPUT_DIR / "chart4_composition_variability.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 4 saved: {output_path}")

    return fig


def chart5_fleet_size_reduction_vs_mcv_adoption(df_full, scv_baseline):
    """
    Chart 5: Scatter plot showing relationship between MCV share and fleet size reduction.
    """
    # Merge with baseline to compute fleet size reduction
    baseline_vehicles = scv_baseline[["instance", "total_vehicles"]].rename(
        columns={"total_vehicles": "scv_baseline_vehicles"}
    )

    df_analysis = df_full.merge(baseline_vehicles, on="instance", how="left")
    df_analysis["fleet_reduction_pct"] = (
        (df_analysis["scv_baseline_vehicles"] - df_analysis["total_vehicles"])
        / df_analysis["scv_baseline_vehicles"]
        * 100
    )

    # Classify by cost regime
    def cost_regime(row):
        if row["alpha"] <= 1.2 and row["C"] <= 20:
            return "Low cost"
        elif row["alpha"] >= 1.6 or row["C"] >= 40:
            return "High cost"
        else:
            return "Medium cost"

    df_analysis["cost_regime"] = df_analysis.apply(cost_regime, axis=1)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.02,
        wspace=0.02,
    )

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])

    regime_colors = {
        "Low cost": "#27ae60",
        "Medium cost": "#f39c12",
        "High cost": "#e74c3c",
    }

    for regime in ["Low cost", "Medium cost", "High cost"]:
        subset = df_analysis[df_analysis["cost_regime"] == regime]
        ax_main.scatter(
            subset["mcv_share"] * 100,
            subset["fleet_reduction_pct"],
            c=regime_colors[regime],
            label=regime,
            alpha=0.5,
            s=20,
        )

    # Add trend line (LOWESS-style with numpy polyfit)
    x = df_analysis["mcv_share"].values * 100
    y = df_analysis["fleet_reduction_pct"].values

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) > 10:
        # Polynomial fit
        z = np.polyfit(x_clean, y_clean, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 100, 100)
        y_trend = p(x_trend)
        ax_main.plot(x_trend, y_trend, "b-", linewidth=3, label="Trend (quadratic fit)")

        # Compute correlation
        corr, p_value = pearsonr(x_clean, y_clean)
        ax_main.text(
            0.05,
            0.95,
            f"Correlation: r={corr:.3f}, p<0.001"
            if p_value < 0.001
            else f"Correlation: r={corr:.3f}, p={p_value:.3f}",
            transform=ax_main.transAxes,
            fontsize=11,
            weight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax_main.set_xlabel("MCV Share (%)", fontsize=12, weight="bold")
    ax_main.set_ylabel(
        "Fleet Size Reduction vs SCV Baseline (%)", fontsize=12, weight="bold"
    )
    ax_main.legend(loc="lower right", fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Top histogram (MCV share distribution)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(
        df_analysis["mcv_share"] * 100,
        bins=50,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax_top.set_ylabel("Frequency", fontsize=10)
    ax_top.tick_params(labelbottom=False)

    # Right histogram (fleet reduction distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(
        df_analysis["fleet_reduction_pct"],
        bins=50,
        orientation="horizontal",
        color="lightcoral",
        edgecolor="black",
        alpha=0.7,
    )
    ax_right.set_xlabel("Frequency", fontsize=10)
    ax_right.tick_params(labelleft=False)

    fig.suptitle(
        "Fleet Size Reduction vs MCV Adoption\n"
        + "Each point = one (demand day, α, C) observation",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    # Save
    output_path = OUTPUT_DIR / "chart5_fleet_size_vs_mcv_adoption.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 5 saved: {output_path}")

    return fig


def chart6b_multigood_by_cost_level(df_full):
    """
    Chart 6B: Grouped bar chart showing effect of multi-good % across different cost levels.
    """
    # Filter to sweet spot
    sweet_spot = df_full[
        (df_full["alpha"] >= 1.2)
        & (df_full["alpha"] < 2.0)
        & (df_full["C"] >= 20)
        & (df_full["C"] < 50)
    ].copy()

    # Create cost levels
    sweet_spot["cost_level"] = pd.cut(
        sweet_spot["alpha"],
        bins=[1.2, 1.4, 1.6, 2.0],
        labels=["1.2≤α<1.4", "1.4≤α<1.6", "1.6≤α<2.0"],
    )

    # Split by median
    median_mg = sweet_spot["multi_good_pct"].median()
    sweet_spot["mg_group"] = sweet_spot["multi_good_pct"].apply(
        lambda x: "Low" if x < median_mg else "High"
    )

    # Compute statistics
    from scipy.stats import ttest_ind

    results = []
    for level in ["1.2≤α<1.4", "1.4≤α<1.6", "1.6≤α<2.0"]:
        level_data = sweet_spot[sweet_spot["cost_level"] == level]
        low = level_data[level_data["mg_group"] == "Low"]["mcv_share"]
        high = level_data[level_data["mg_group"] == "High"]["mcv_share"]

        if len(low) > 0 and len(high) > 0:
            delta = high.mean() - low.mean()
            t_stat, p_val = ttest_ind(high, low)

            results.append(
                {
                    "level": level,
                    "low_mean": low.mean(),
                    "high_mean": high.mean(),
                    "low_se": low.sem(),
                    "high_se": high.sem(),
                    "delta": delta,
                    "p_val": p_val,
                    "n_low": len(low),
                    "n_high": len(high),
                }
            )

    results_df = pd.DataFrame(results)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(results_df))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        results_df["low_mean"],
        width,
        label=f"Low Multi-Good (<{median_mg:.1%})",
        color="#3498db",
        edgecolor="black",
        linewidth=1.5,
        yerr=results_df["low_se"],
        capsize=5,
        error_kw={"linewidth": 2},
    )

    bars2 = ax.bar(
        x + width / 2,
        results_df["high_mean"],
        width,
        label=f"High Multi-Good (≥{median_mg:.1%})",
        color="#2ecc71",
        edgecolor="black",
        linewidth=1.5,
        yerr=results_df["high_se"],
        capsize=5,
        error_kw={"linewidth": 2},
    )

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1 + 0.02,
            f"{height1:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2 + 0.02,
            f"{height2:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Add difference annotations
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        sig = "***" if row["p_val"] < 0.001 else "**" if row["p_val"] < 0.01 else "*"
        y_pos = float(max(row["low_mean"], row["high_mean"])) + 0.12
        ax.text(
            i,
            y_pos,
            f"Δ = {row['delta']:+.1%}{sig}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

        # Add sample sizes
        ax.text(
            i,
            -0.08,
            f"n={int(row['n_low'])}/{int(row['n_high'])}",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
        )

    # Formatting
    ax.set_xlabel("MCV Cost Level (α range)", fontsize=13, weight="bold")
    ax.set_ylabel("Mean MCV Share", fontsize=13, weight="bold")
    ax.set_title(
        "Effect of Demand Heterogeneity on MCV Adoption Across Cost Levels\nSweet Spot: 1.2≤α<2.0, 20%≤C<50%",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["level"], fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_ylim(-0.1, max(results_df["high_mean"]) + 0.2)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "chart6b_multigood_by_cost_level.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 6B (Multi-Good by Cost Level) saved: {output_path}")

    return fig


def chart6c_multigood_tercile_bars(df_full):
    """
    Chart 6C: Simple tercile bar chart showing monotonic increase in MCV adoption.
    """
    # Filter to sweet spot
    sweet_spot = df_full[
        (df_full["alpha"] >= 1.2)
        & (df_full["alpha"] < 2.0)
        & (df_full["C"] >= 20)
        & (df_full["C"] < 50)
    ].copy()

    # Create terciles
    sweet_spot["mg_tercile"] = pd.qcut(
        sweet_spot["multi_good_pct"], q=3, labels=["Low", "Medium", "High"]
    )

    # Compute statistics
    tercile_stats = []
    for tercile in ["Low", "Medium", "High"]:
        data = sweet_spot[sweet_spot["mg_tercile"] == tercile]
        tercile_stats.append(
            {
                "tercile": tercile,
                "mg_range": f"{data['multi_good_pct'].min():.1%}-{data['multi_good_pct'].max():.1%}",
                "mean_mcv": data["mcv_share"].mean(),
                "median_mcv": data["mcv_share"].median(),
                "se": data["mcv_share"].sem(),
                "n": len(data),
            }
        )

    stats_df = pd.DataFrame(tercile_stats)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color gradient
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    # Plot bars
    bars = ax.bar(
        range(3),
        stats_df["mean_mcv"],
        color=colors,
        edgecolor="black",
        linewidth=2,
        yerr=stats_df["se"],
        capsize=8,
        error_kw={"linewidth": 2},
    )

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, stats_df.itertuples())):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )

        # Add tercile info below
        ax.text(
            i, -0.15, f"{row.mg_range}\n(N={row.n})", ha="center", va="top", fontsize=10
        )

    # Add arrows and deltas
    # Low to Medium
    delta1 = stats_df.iloc[1]["mean_mcv"] - stats_df.iloc[0]["mean_mcv"]
    ax.annotate(
        "",
        xy=(1, stats_df.iloc[1]["mean_mcv"]),
        xytext=(0, stats_df.iloc[0]["mean_mcv"]),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    ax.text(
        0.5,
        (stats_df.iloc[0]["mean_mcv"] + stats_df.iloc[1]["mean_mcv"]) / 2 + 0.05,
        f"+{delta1:.1%}",
        ha="center",
        fontsize=11,
        weight="bold",
    )

    # Medium to High
    delta2 = stats_df.iloc[2]["mean_mcv"] - stats_df.iloc[1]["mean_mcv"]
    ax.annotate(
        "",
        xy=(2, stats_df.iloc[2]["mean_mcv"]),
        xytext=(1, stats_df.iloc[1]["mean_mcv"]),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    ax.text(
        1.5,
        (stats_df.iloc[1]["mean_mcv"] + stats_df.iloc[2]["mean_mcv"]) / 2 + 0.05,
        f"+{delta2:.1%}",
        ha="center",
        fontsize=11,
        weight="bold",
    )

    # Total delta
    total_delta = stats_df.iloc[2]["mean_mcv"] - stats_df.iloc[0]["mean_mcv"]
    from scipy.stats import ttest_ind

    low_data = sweet_spot[sweet_spot["mg_tercile"] == "Low"]["mcv_share"]
    high_data = sweet_spot[sweet_spot["mg_tercile"] == "High"]["mcv_share"]
    t_stat, p_val = ttest_ind(high_data, low_data)

    ax.text(
        0.5,
        0.98,
        f"Total Effect: {total_delta:+.1%} (p < 0.001***)\nLow → High Multi-Good Customer Proportion",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", edgecolor="black", linewidth=2
        ),
    )

    # Formatting
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["Low\nMulti-Good", "Medium\nMulti-Good", "High\nMulti-Good"],
        fontsize=13,
        weight="bold",
    )
    ax.set_ylabel("Mean MCV Share", fontsize=13, weight="bold")
    ax.set_title(
        "MCV Adoption Increases with Multi-Good Customer Proportion\nSweet Spot: 1.2≤α<2.0, 20%≤C<50%",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_ylim(-0.2, max(stats_df["mean_mcv"]) + 0.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "chart6c_multigood_terciles.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart 6C (Multi-Good Terciles) saved: {output_path}")

    return fig


def generate_summary_statistics(metrics, df_full, regime_dist):
    """Generate summary statistics table."""

    summary = {
        "Total observations": len(df_full),
        "Unique demand days": df_full["instance"].nunique(),
        "Parameter combinations": len(metrics),
        "Alpha range": f"{df_full['alpha'].min():.1f} - {df_full['alpha'].max():.1f}",
        "C range": f"{df_full['C'].min():.0f}% - {df_full['C'].max():.0f}%",
        "Multi-good % (mean)": f"{df_full['multi_good_pct'].mean():.1%}",
        "Multi-good % (range)": f"{df_full['multi_good_pct'].min():.1%} - {df_full['multi_good_pct'].max():.1%}",
        "Mean MCV share": f"{df_full['mcv_share'].mean():.1%}",
        "Median MCV share": f"{df_full['mcv_share'].median():.1%}",
        "Days with MCV > 0": f"{(df_full['mcv_vehicles'] > 0).sum()} / {len(df_full)} ({(df_full['mcv_vehicles'] > 0).mean():.1%})",
        "Pure MCV days (≥99%)": f"{(df_full['mcv_share'] >= 0.99).sum()} / {len(df_full)} ({(df_full['mcv_share'] >= 0.99).mean():.1%})",
        "True mixed days (10-90%)": f"{((df_full['mcv_share'] >= 0.1) & (df_full['mcv_share'] <= 0.9)).sum()} / {len(df_full)} ({((df_full['mcv_share'] >= 0.1) & (df_full['mcv_share'] <= 0.9)).mean():.1%})",
        "Mean fleet size": f"{df_full['total_vehicles'].mean():.1f}",
        "Mean MCV vehicles": f"{df_full['mcv_vehicles'].mean():.1f}",
        "Mean SCV vehicles": f"{df_full['scv_vehicles'].mean():.1f}",
    }

    print("\n" + "=" * 60)
    print("FLEET COMPOSITION ANALYSIS - SUMMARY STATISTICS")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key:.<40} {value}")
    print("=" * 60)

    # Save to file
    summary_path = OUTPUT_DIR / "summary_statistics.txt"
    with open(summary_path, "w") as f:
        f.write("FLEET COMPOSITION ANALYSIS - SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ Summary statistics saved: {summary_path}")

    return summary


def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("FLEET COMPOSITION ANALYSIS FOR MIXED FLEET OPTIMIZATION")
    print("=" * 60)

    # Load data
    print("\n[1/9] Loading data...")
    mixed, scv_baseline = load_mixed_results()

    # Compute metrics
    print("\n[2/9] Computing composition metrics...")
    metrics, regime_dist, df_full = compute_composition_metrics(mixed)

    # Generate charts
    print("\n[3/9] Generating Chart 1: MCV Adoption Landscape...")
    chart1_mcv_adoption_landscape(metrics)

    print("\n[4/9] Generating Chart 2: Fleet Composition Regimes...")
    chart2_fleet_composition_regimes(regime_dist, metrics)

    print("\n[5/9] Generating Chart 3: MCV Adoption Probability Surface...")
    chart3_mcv_adoption_probability_surface(metrics)

    print("\n[6/9] Generating Chart 4: Fleet Composition Variability...")
    chart4_fleet_composition_variability(df_full)

    print("\n[7/9] Generating Chart 5: Fleet Size Reduction vs MCV Adoption...")
    chart5_fleet_size_reduction_vs_mcv_adoption(df_full, scv_baseline)

    print("\n[8/9] Generating Chart 6B: Multi-Good by Cost Level...")
    chart6b_multigood_by_cost_level(df_full)

    print("\n[9/9] Generating Chart 6C: Multi-Good Terciles...")
    chart6c_multigood_tercile_bars(df_full)

    # Generate summary statistics
    print("\n[Summary] Generating summary statistics...")
    generate_summary_statistics(metrics, df_full, regime_dist)

    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE!")
    print(f"✓ All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

    plt.close("all")


if __name__ == "__main__":
    main()
