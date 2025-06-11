"""
data_processing.py

Light-weight helpers for locating and loading demand CSVs.  The idea is to
keep any filesystem assumptions (paths, encodings, column renames) away from
business logic.  This way, the core optimisation code can operate on clean
DataFrames without ever touching `Path` or `pd.read_csv`.

If tomorrow the sales team decides to move files or change headers, you only
need to patch this module.
"""

from pathlib import Path

import pandas as pd

from fleetmix.utils.logging import log_detail


def data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def get_demand_profiles_dir() -> Path:
    """Get the demand profiles directory path."""
    return data_dir() / "demand_profiles"


def load_customer_demand(demand_file: str):
    """Load customer demand from CSV file.

    Args:
        demand_file: Either a filename relative to demand_profiles directory,
                    or an absolute/relative path to a CSV file.
    """
    demand_path = Path(demand_file)

    if demand_path.is_absolute() or demand_path.exists():
        # It's an absolute path or relative path that exists from current directory
        csv_file_path = demand_path
    else:
        # Treat as filename relative to demand_profiles directory
        csv_file_path = get_demand_profiles_dir() / demand_file

    log_detail(f"Loading customer demand from {csv_file_path}")

    # Read CSV with existing headers
    df = pd.read_csv(
        csv_file_path,
        dtype={
            "ClientID": str,
            "Lat": float,
            "Lon": float,
            "Kg": int,
            "ProductType": str,
        },
        encoding="latin-1",
    )

    # Rename columns to match our expected format
    df = df.rename(
        columns={
            "ClientID": "Customer_ID",
            "Lat": "Latitude",
            "Lon": "Longitude",
            "Kg": "Units_Demand",
            "ProductType": "Demand_Type",
        }
    )

    # Create pivot table and fill NaN values with 0
    df_pivot = (
        df.pivot_table(
            index=["Customer_ID", "Latitude", "Longitude"],
            columns="Demand_Type",
            values="Units_Demand",
            fill_value=0,
            aggfunc="sum",
        )
        .fillna(0)
        .reset_index()
    )

    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(
        columns={
            "Dry": "Dry_Demand",
            "Chilled": "Chilled_Demand",
            "Frozen": "Frozen_Demand",
        }
    )

    # Ensure all demand columns are integers
    demand_cols = ["Dry_Demand", "Chilled_Demand", "Frozen_Demand"]
    for col in demand_cols:
        df_pivot[col] = df_pivot[col].astype(int)

    # Set zero demand to 1 if all demands are zero
    if (df_pivot[demand_cols] == 0).all(axis=1).any():
        df_pivot.loc[
            (df_pivot["Dry_Demand"] == 0)
            & (df_pivot["Chilled_Demand"] == 0)
            & (df_pivot["Frozen_Demand"] == 0),
            "Dry_Demand",
        ] = 1

    return df_pivot
