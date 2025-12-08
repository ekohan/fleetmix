"""
Generate MCV Adoption Landscape Heatmap and Summary Table.

This script loads raw JSON results from a directory and:
1. Generates the MCV adoption landscape heatmap (Chart 1).
2. Generates a summary table for alpha=1.6 showing impacts of compartment cost.
3. Generates a runtime analysis table showing Mean, P95, and Max times.
4. Creates a README.md with the tables and plot.

Usage:
    python generate_section_7_3.py /path/to/results/directory
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MCV Adoption Landscape Heatmap and Summary Table."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing raw JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the output plot and README (defaults to input directory)",
    )
    return parser.parse_args()


def load_from_json_files(results_dir: Path) -> pd.DataFrame:
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        return pd.DataFrame()

    print(f"Found {len(json_files)} JSON files. Loading...")
    results: List[Dict[str, Any]] = []

    for json_path in tqdm(json_files, desc="Loading JSONs"):
        with open(json_path, "r") as f:
            results.append(json.load(f))

    return pd.DataFrame(results)


def get_global_wall_time(measurements: Any) -> float:
    if not isinstance(measurements, list):
        return np.nan
    for m in measurements:
        if isinstance(m, dict) and m.get("span_name") == "global":
            return float(m.get("wall_time", np.nan))
    return np.nan


def load_data(results_dir: Path) -> pd.DataFrame:
    print(f"Searching for raw JSON files in {results_dir}...")
    df = load_from_json_files(results_dir)

    if df.empty:
        raise FileNotFoundError(f"Could not find valid raw JSON files in {results_dir}")

    if "fleet_type" in df.columns:
        mixed = df[df["fleet_type"] == "MIXED"].copy()
        if not mixed.empty:
            print(f"Filtered for MIXED fleet type. Rows: {len(mixed)}")
            df = mixed

    required_cols = [
        "alpha",
        "C",
        "mcv_share",
        "mcv_vehicles",
        "total_cost",
        "total_vehicles",
        "average_visits_per_customer",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    if "time_measurements" in df.columns:
        df["global_runtime"] = df["time_measurements"].apply(get_global_wall_time)
    else:
        df["global_runtime"] = np.nan

    print(f"Loaded {len(df)} rows.")
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing aggregation metrics...")

    df["alpha"] = pd.to_numeric(df["alpha"])
    df["C"] = pd.to_numeric(df["C"])

    metrics = (
        df.groupby(["alpha", "C"])
        .agg(
            {
                "mcv_share": ["mean"],
                "mcv_vehicles": [
                    (
                        "adoption_days",
                        lambda x: (x > 0).sum(),
                    ),
                    (
                        "pure_mcv_days",
                        lambda x: (df.loc[x.index, "mcv_share"] >= 0.99).sum(),
                    ),
                ],
                "instance": "count",
            }
        )
        .reset_index()
    )

    metrics.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in metrics.columns.values
    ]

    return metrics


def plot_heatmap(metrics: pd.DataFrame, output_dir: Path) -> None:
    print("Generating heatmap...")

    if metrics.empty:
        print("Error: No metrics to plot.")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(12, 10))

    pivot_share = metrics.pivot(index="alpha", columns="C", values="mcv_share_mean")
    pivot_adoption = metrics.pivot(
        index="alpha", columns="C", values="mcv_vehicles_adoption_days"
    )
    pivot_pure_mcv = metrics.pivot(
        index="alpha", columns="C", values="mcv_vehicles_pure_mcv_days"
    )

    im = ax.imshow(
        pivot_share.values * 100,
        cmap="YlGnBu",
        aspect="auto",
        vmin=0,
        vmax=100,
        origin="upper",
    )

    for i in range(len(pivot_share.index)):
        for j in range(len(pivot_share.columns)):
            share_pct = pivot_share.iloc[i, j] * 100
            adoption_days = int(pivot_adoption.iloc[i, j])
            pure_days = int(pivot_pure_mcv.iloc[i, j])

            text_color = "white" if share_pct > 50 else "black"
            text = f"{share_pct:.0f}%\n{adoption_days}/70\n({pure_days}$\\star$)"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
                fontfamily="sans-serif",
            )

    ax.set_xticks(np.arange(len(pivot_share.columns)))
    ax.set_yticks(np.arange(len(pivot_share.index)))
    ax.set_xticklabels([f"{int(c)}%" for c in pivot_share.columns])
    ax.set_yticklabels(
        [f"{a:.1f}" if a != int(a) else f"{int(a)}.0" for a in pivot_share.index]
    )

    ax.set_xlabel("Compartment Setup Factor (c, % of SCV fixed cost)", fontsize=12)
    ax.set_ylabel(r"Premium Factor ($\alpha$)", fontsize=12)
    ax.set_title(
        r"Values: MCV Share (%) | Adoption Days / 70 | (Pure MCV Days $\star$)",
        fontsize=11,
        pad=12,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average MCV Share (%)", fontsize=11)

    ax.set_xticks(np.arange(len(pivot_share.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_share.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    output_path_pdf = output_dir / "mcv_adoption_heatmap.pdf"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    print(f"\n✓ Heatmap saved: {output_path_pdf}")

    output_path_png = output_dir / "mcv_adoption_heatmap.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved: {output_path_png}")

    plt.close(fig)


def generate_table_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerating summary table for α=1.6...")

    mask_alpha = np.isclose(df["alpha"], 1.6)
    df_subset = df[mask_alpha].copy()

    if df_subset.empty:
        print("Warning: No data found for alpha=1.6. Skipping table generation.")
        return pd.DataFrame()

    # Use all available C values for the table
    summary = (
        df_subset.groupby("C")
        .agg(
            {
                "mcv_share": "mean",
                "total_cost": "mean",
                "total_vehicles": "mean",
                "average_visits_per_customer": "mean",
            }
        )
        .reset_index()
    )

    summary.columns = [
        "Compartment setup factor (c)",
        "MCV share",
        "Total cost",
        "Fleet size",
        "Visits per customer",
    ]

    if summary["Compartment setup factor (c)"].max() > 1.0:
        summary["Compartment setup factor (c)"] = summary[
            "Compartment setup factor (c)"
        ].apply(lambda x: x / 100.0)

    return summary


def generate_runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerating runtime analysis...")

    if "global_runtime" not in df.columns or df["global_runtime"].isna().all():
        print("No runtime data available.")
        return pd.DataFrame()

    runtimes = df["global_runtime"].dropna()

    stats = {
        "Metric": ["Mean", "P95", "Max"],
        "Time (s)": [runtimes.mean(), runtimes.quantile(0.95), runtimes.max()],
    }

    return pd.DataFrame(stats)


def create_readme(
    table_df: pd.DataFrame, runtime_df: pd.DataFrame, output_dir: Path
) -> None:
    readme_path = output_dir / "README.md"

    markdown_content = "# Fleet Composition Analysis Results\n\n"
    markdown_content += "## MCV Adoption Landscape\n\n"
    markdown_content += "![MCV Adoption Landscape](mcv_adoption_heatmap.png)\n\n"
    markdown_content += "## Impacts of Cost Structure (alpha = 1.6)\n\n"

    if not table_df.empty:
        formatted_df = table_df.copy()
        formatted_df["MCV share"] = formatted_df["MCV share"].apply(
            lambda x: f"{x:.0%}"
        )
        formatted_df["Total cost"] = formatted_df["Total cost"].apply(
            lambda x: f"{x:,.0f}"
        )
        formatted_df["Fleet size"] = formatted_df["Fleet size"].apply(
            lambda x: f"{x:.1f}"
        )
        formatted_df["Visits per customer"] = formatted_df["Visits per customer"].apply(
            lambda x: f"{x:.1f}"
        )
        formatted_df["Compartment setup factor (c)"] = formatted_df[
            "Compartment setup factor (c)"
        ].apply(lambda x: f"{x:.1f}")

        markdown_content += formatted_df.to_markdown(index=False)
        markdown_content += "\n\n"
    else:
        markdown_content += "No data available for the summary table (alpha=1.6).\n\n"

    markdown_content += "## Runtime Analysis\n\n"

    if not runtime_df.empty:
        formatted_runtime = runtime_df.copy()
        formatted_runtime["Time (s)"] = formatted_runtime["Time (s)"].apply(
            lambda x: f"{x:.2f}"
        )
        markdown_content += formatted_runtime.to_markdown(index=False)
        markdown_content += "\n"
    else:
        markdown_content += "No runtime data available.\n"

    with open(readme_path, "w") as f:
        f.write(markdown_content)

    print(f"\n✓ README.md created: {readme_path}")


def main():
    args = parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(results_dir)

    metrics = compute_metrics(df)
    plot_heatmap(metrics, output_dir)

    table_df = generate_table_data(df)

    runtime_df = generate_runtime_table(df)

    create_readme(table_df, runtime_df, output_dir)


if __name__ == "__main__":
    main()
