"""Dataset characterization for Alpha & C experiments.

This script computes summary statistics for each daily demand instance (CSV file)
and generates exploratory plots to understand the scale and variability of the
experimental dataset.  The insights can help contextualise the fleet-mix results
and guide interpretation of the economic analyses.

Run as a standalone module:

    python -m experiments.alpha_analysis.dataset_characterization

Outputs are written to  results/demand_characterization/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, KDTree

# -----------------------------------------------------------------------------
# Constants – depot location (read from YAML if available)
# -----------------------------------------------------------------------------

DEPOT_LAT, DEPOT_LON = 4.7, -74.1  # degrees (from default_config.yaml)

# Earth radius for haversine distance (km)
R_EARTH_KM = 6371.0

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Directory containing the raw daily demand CSV files
DATA_DIR = Path("src/fleetmix/benchmarking/datasets/case")

# Output directory for figures and summary tables
OUTPUT_DIR = Path("results/demand_characterization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "daily_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "dataset_overview.json"

sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 150})

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _summarise_day(csv_path: Path) -> dict[str, float]:
    """Return basic statistics for a single demand CSV file."""
    df = pd.read_csv(csv_path)

    # Defensive checks
    if "Kg" not in df.columns:
        raise ValueError(f"Expected column 'Kg' in {csv_path}")

    day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
    total_kg = df["Kg"].sum()
    num_customers = len(df)

    return {
        "day_id": day_id,
        "date": pd.to_datetime(day_id, errors="coerce"),
        "num_customers": num_customers,
        "total_kg": total_kg,
        "mean_kg": df["Kg"].mean(),
        "median_kg": df["Kg"].median(),
        "std_kg": df["Kg"].std(),
        "min_kg": df["Kg"].min(),
        "p10_kg": np.percentile(df["Kg"], 10),
        "p90_kg": np.percentile(df["Kg"], 90),
        "max_kg": df["Kg"].max(),
    }


def _aggregate_daily_stats() -> pd.DataFrame:
    """Parse all CSV files and return per-day statistics as a DataFrame."""
    records: list[dict[str, float]] = []
    csv_files = sorted(DATA_DIR.glob("*_demand.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No demand CSV files found in {DATA_DIR}")

    for path in csv_files:
        try:
            records.append(_summarise_day(path))
        except Exception as err:
            print(f"Skipping {path.name}: {err}")

    df = pd.DataFrame.from_records(records)
    # Sort by date when available
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _histogram(data: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, logx: bool = False):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True, bins=30)
    if logx:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x, y=y)
    sns.regplot(x=x, y=y, scatter=False, color="red", ci=None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("DEMAND DATASET CHARACTERISATION")
    print("Reading demand CSVs from", DATA_DIR.resolve())

    daily_df = _aggregate_daily_stats()
    daily_df.to_csv(SUMMARY_CSV, index=False)

    # ------------------------------------------------------------------
    # Dataset-level overview (printed + JSON for consumption elsewhere)
    # ------------------------------------------------------------------
    overview = {
        "num_days": len(daily_df),
        "customers": {
            "min": int(daily_df["num_customers"].min()),
            "max": int(daily_df["num_customers"].max()),
            "mean": float(daily_df["num_customers"].mean()),
            "median": float(daily_df["num_customers"].median()),
        },
        "total_kg": {
            "min": float(daily_df["total_kg"].min()),
            "max": float(daily_df["total_kg"].max()),
            "mean": float(daily_df["total_kg"].mean()),
            "median": float(daily_df["total_kg"].median()),
        },
    }

    with open(SUMMARY_JSON, "w") as fp:
        json.dump(overview, fp, indent=2)

    print(json.dumps(overview, indent=2))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _histogram(
        daily_df["num_customers"],
        "Distribution of Daily Customer Count",
        "Number of Customers",
        "Number of Days",
        "hist_num_customers.png",
    )

    _histogram(
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

    # Aggregate per-customer demand across all days for a finer-grained view
    per_customer_kg = []
    for path in DATA_DIR.glob("*_demand.csv"):
        df = pd.read_csv(path)
        per_customer_kg.extend(df["Kg"].tolist())

    _histogram(
        pd.Series(per_customer_kg),
        "Distribution of Demand per Customer",
        "kg per drop",
        "Number of Customer Deliveries",
        "hist_kg_per_customer.png",
        logx=True,
    )

    # ------------------------------------------------------------------
    #  Stage 1 – Spatial dispersion (distance to depot)
    # ------------------------------------------------------------------

    def _haversine(lat1, lon1):
        """Vectorised haversine distance in km to depot."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = math.radians(DEPOT_LAT), math.radians(DEPOT_LON)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))

    spatial_records: list[pd.DataFrame] = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        df["distance_km"] = _haversine(df["Lat"], df["Lon"])
        df["day_id"] = day_id
        spatial_records.append(df[["day_id", "distance_km", "Lat", "Lon"]])

    all_spatial = pd.concat(spatial_records, ignore_index=True)

    # Histogram of distance
    _histogram(
        all_spatial["distance_km"],
        "Customer Distance to Depot",
        "Distance (km)",
        "Number of Customer Deliveries",
        "hist_distance_km.png",
    )

    # KDE heat-map
    plt.figure(figsize=(6, 6))
    sns.kdeplot(
        x=all_spatial["Lon"],
        y=all_spatial["Lat"],
        cmap="viridis",
        shade=True,
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
    plt.savefig(OUTPUT_DIR / "kde_spatial_density.png")
    plt.close()

    # Per-day distance stats
    dist_stats = (
        all_spatial.groupby("day_id")["distance_km"]
        .agg(["mean", "median", "std", "max", "count"])
        .reset_index()
    )
    dist_stats.to_csv(OUTPUT_DIR / "distance_stats_per_day.csv", index=False)

    # Overall distance overview
    distance_overview = {
        "min_km": float(all_spatial["distance_km"].min()),
        "max_km": float(all_spatial["distance_km"].max()),
        "mean_km": float(all_spatial["distance_km"].mean()),
        "median_km": float(all_spatial["distance_km"].median()),
        "p95_km": float(np.percentile(all_spatial["distance_km"], 95)),
    }
    with open(OUTPUT_DIR / "distance_overview.json", "w") as fp:
        json.dump(distance_overview, fp, indent=2)

    # ------------------------------------------------------------------
    #  Stage 2 – Temporal patterns
    # ------------------------------------------------------------------

    daily_df["weekday"] = daily_df["date"].dt.day_name()
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    def _weekday_box(col: str, fname: str, ylabel: str):
        plt.figure(figsize=(7, 4))
        sns.boxplot(
            x="weekday", y=col, data=daily_df, order=weekday_order, color="lightsteelblue"
        )
        plt.ylabel(ylabel)
        plt.xlabel("Weekday")
        plt.xticks(rotation=45)
        plt.title(f"Daily {ylabel} by Weekday")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname)
        plt.close()

    _weekday_box("total_kg", "box_total_kg_weekday.png", "Total kg")
    _weekday_box("num_customers", "box_customers_weekday.png", "Number of Customers")

    # Manual autocorrelation (up to 30 lags) - REMOVED

    temporal_metrics = {
        "peak_to_average_total_kg": float(daily_df["total_kg"].max() / daily_df["total_kg"].mean()),
        "peak_to_average_customers": float(
            daily_df["num_customers"].max() / daily_df["num_customers"].mean()
        ),
    }
    with open(OUTPUT_DIR / "temporal_metrics.json", "w") as fp:
        json.dump(temporal_metrics, fp, indent=2)

    # STAGES 4 and 5 REMOVED

    # ------------------------------------------------------------------
    #  Stage 3 – Correlation heat-map & Table 1 synthesis (was Stage 6)
    # ------------------------------------------------------------------

    # Merge day-level frames: base daily_df, distance stats
    dist_daily = dist_stats.rename(columns={
        "mean": "mean_distance_km",
        "median": "median_distance_km",
        "std": "std_distance_km",
        "max": "max_distance_km",
        "count": "num_drops",  # same as customers; kept for completeness
    })[[
        "day_id",
        "mean_distance_km",
        "max_distance_km",
    ]]

    corr_df = daily_df.merge(dist_daily, on="day_id")
    
    # Base numeric cols for original table
    table1_numeric_cols = [
        "num_customers",
        "total_kg",
        "mean_kg",
        "std_kg",
        "mean_distance_km",
        "max_distance_km",
    ]

    # Table 1 – descriptive stats
    table_stats = corr_df[table1_numeric_cols].agg(["min", "mean", "median", "max", "std"]).T
    table_stats["cv"] = table_stats["std"] / table_stats["mean"]
    table_stats.to_csv(OUTPUT_DIR / "table1_daily_stats.csv")

    # ------------------------------------------------------------------
    #  Stage 4: Appendix Figures & Advanced Metrics (was Stage 7)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING APPENDIX FIGURES")

    # --- Geographic service area & demand density ---
    geo_records = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df_day = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        coords = df_day[["Lon", "Lat"]].to_numpy()
        
        area_sq_km = 0
        if len(coords) >= 3:
            try:
                # Note: This is a planar projection approximation
                hull = ConvexHull(coords)
                area_sq_km = hull.volume # area for 2D
            except Exception: # qhull error
                pass 
        
        geo_records.append({
            "day_id": day_id,
            "geo_area_sq_km": area_sq_km,
            "demand_density": len(df_day) / area_sq_km if area_sq_km > 0 else 0,
        })
    geo_df = pd.DataFrame(geo_records)
    corr_df = corr_df.merge(geo_df, on="day_id")
    
    # --- Average distance between customers ---
    dist_between_cust_records = []
    for csv_path in DATA_DIR.glob("*_demand.csv"):
        day_id = csv_path.stem.replace("sales_", "").replace("_demand", "")
        df_day = pd.read_csv(csv_path, usecols=["Lat", "Lon"])
        if len(df_day) > 1:
            coords_rad = np.radians(df_day[['Lat', 'Lon']].to_numpy())
            tree = KDTree(coords_rad)
            # Query each point for its nearest neighbor
            distances, _ = tree.query(coords_rad, k=2)
            # Distances are in radians, convert to km
            avg_dist_km = np.mean(distances[:, 1]) * R_EARTH_KM
        else:
            avg_dist_km = 0
            
        dist_between_cust_records.append({
            "day_id": day_id,
            "avg_dist_between_customers_km": avg_dist_km
        })
    dist_between_df = pd.DataFrame(dist_between_cust_records)
    corr_df = corr_df.merge(dist_between_df, on="day_id")
    
    # Update numeric cols for correlation matrix
    extended_numeric_cols = table1_numeric_cols + ["geo_area_sq_km", "demand_density", "avg_dist_between_customers_km"]

    corr_matrix = corr_df[extended_numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation Matrix of Day-Level Features (Extended)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_correlations_extended.png")
    plt.close()

    # --- Figure A1: Temporal Demand Patterns ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    # Panel A: Daily volume
    axes[0].plot(daily_df["date"], daily_df["total_kg"], marker='o', linestyle='-', markersize=4)
    axes[0].set_title("Panel A: Daily Total Demand Volume")
    axes[0].set_ylabel("Total kg")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Panel B: Day-of-week boxplots
    sns.boxplot(x="weekday", y="total_kg", data=daily_df, order=weekday_order, ax=axes[1])
    axes[1].set_title("Panel B: Day-of-Week Effects on Total Demand")
    axes[1].set_xlabel("Weekday")
    axes[1].set_ylabel("Total kg")

    # Panel C: Customer count
    axes[2].plot(daily_df["date"], daily_df["num_customers"], marker='o', linestyle='-', markersize=4, color='teal')
    axes[2].set_title("Panel C: Daily Customer Count")
    axes[2].set_ylabel("Number of Customers")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.suptitle("Figure A1: Temporal Demand Patterns", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_A1_temporal_patterns.png")
    plt.close()

    # --- Figure A2: Demand Scale and Distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(daily_df["num_customers"], kde=True, ax=axes[0])
    axes[0].set_title("Panel A: Daily Customer Counts")
    axes[0].set_xlabel("Number of Customers")
    
    sns.histplot(daily_df["total_kg"], kde=True, ax=axes[1])
    axes[1].set_title("Panel B: Daily Total Volume")
    axes[1].set_xlabel("Total kg")

    sns.histplot(corr_df["geo_area_sq_km"], kde=True, ax=axes[2])
    axes[2].set_title("Panel C: Geographic Coverage per Day")
    axes[2].set_xlabel("Service Area (km²)")
    fig.suptitle("Figure A2: Demand Scale and Distribution", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_A2_demand_scale.png")
    plt.close()

    # --- Figure A3: Demand Variability and Complexity ---
    corr_df["cv_total_kg"] = corr_df["std_kg"] / corr_df["mean_kg"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(corr_df["date"], corr_df["cv_total_kg"], marker='o', linestyle='-', markersize=4)
    axes[0].set_title("Panel A: Coefficient of Variation (Order Size)")
    axes[0].set_ylabel("CV of kg per drop")
    axes[0].tick_params(axis='x', rotation=45)

    sns.scatterplot(data=corr_df, x="geo_area_sq_km", y="demand_density", ax=axes[1])
    axes[1].set_title("Panel B: Demand Density")
    axes[1].set_xlabel("Service Area (km²)")
    axes[1].set_ylabel("Customers per km²")
    fig.suptitle("Figure A3: Demand Variability and Complexity", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_A3_demand_variability.png")
    plt.close()

    # --- Figure A4: Stochastic Demand Summary Statistics (Table) ---
    summary_cols = [
        "num_customers", "total_kg", "mean_kg", "cv_total_kg",
        "mean_distance_km", "avg_dist_between_customers_km", "geo_area_sq_km", 
        "demand_density",
    ]
    table_df = corr_df[summary_cols].copy()
    
    # Round km values
    for col in ["mean_distance_km", "avg_dist_between_customers_km", "geo_area_sq_km"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(0).astype(int)

    table_A4_stats = table_df.agg(['min', 'max', 'mean', 'std']).T
    
    # Prettier labels for the table
    pretty_labels = {
        "num_customers": "Number of Customers",
        "total_kg": "Total Demand (kg)",
        "mean_kg": "Mean Order Size (kg)",
        "cv_total_kg": "CV of Order Size",
        "mean_distance_km": "Mean Distance to Depot (km)",
        "avg_dist_between_customers_km": "Mean Inter-Customer Distance (km)",
        "geo_area_sq_km": "Service Area (km²)",
        "demand_density": "Demand Density (Customers/km²)",
    }
    table_A4_stats = table_A4_stats.rename(index=pretty_labels)
    table_A4_stats.to_csv(OUTPUT_DIR / "table_A4_summary_stats.csv")

    # --- Figure A5: Weekday distribution of customers and demand ---
    _create_daily_distribution_plot(daily_df, OUTPUT_DIR)


    print("Figures saved to", OUTPUT_DIR.resolve())
    print("Characterisation complete ✅")


def _create_daily_distribution_plot(daily_df: pd.DataFrame, output_dir: Path):
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
    
    # Customers by weekday
    sns.boxplot(x="weekday", y="num_customers", data=daily_df, order=weekday_order, ax=axes[0], palette="viridis", showfliers=False)
    sns.stripplot(x="weekday", y="num_customers", data=daily_df, order=weekday_order, ax=axes[0], color=".25", size=5, alpha=0.7)
    axes[0].set_title("Distribution of Daily Customers by Weekday", fontsize=14)
    axes[0].set_xlabel("Weekday", fontsize=12)
    axes[0].set_ylabel("Number of Customers", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Demand by weekday
    sns.boxplot(x="weekday", y="total_kg", data=daily_df, order=weekday_order, ax=axes[1], palette="viridis", showfliers=False)
    sns.stripplot(x="weekday", y="total_kg", data=daily_df, order=weekday_order, ax=axes[1], color=".25", size=5, alpha=0.7)
    axes[1].set_title("Distribution of Daily Demand (kg) by Weekday", fontsize=14)
    axes[1].set_xlabel("Weekday", fontsize=12)
    axes[1].set_ylabel("Total Demand (kg)", fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.suptitle("Daily Customer and Demand Distribution by Weekday", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "fig_A5_weekday_distributions.png")
    plt.close()


if __name__ == "__main__":
    main() 