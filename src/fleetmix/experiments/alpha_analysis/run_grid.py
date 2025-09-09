"""
Grid executor for alpha analysis.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fleetmix.experiments.alpha_analysis.config import (
    ALPHA_GRID,
    C_VALUES,
    DEMAND_FILES,
)
from fleetmix.experiments.alpha_analysis.fleet_templates import (
    make_mcv_fleet,
    make_scv_fleet,
)
from fleetmix.experiments.alpha_analysis.run_day import run_day

PKG_DIR = Path(__file__).resolve().parent
RESULTS_RAW = PKG_DIR / "results" / "raw"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = PKG_DIR / "results" / "summary.parquet"


def convert_numpy_types(obj):
    """Convert numpy types and complex objects to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif is_dataclass(obj):
        # Handle dataclass objects by converting to dict first
        if hasattr(obj, "to_dict"):
            return convert_numpy_types(obj.to_dict())
        else:
            return convert_numpy_types(asdict(obj))  # type: ignore
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(item) for item in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    elif isinstance(obj, set):
        return list(convert_numpy_types(list(obj)))
    elif hasattr(obj, "__dict__"):
        # Handle other complex objects by converting their __dict__
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


def main(config_path: Path | None = None) -> None:
    """
    CLI entry-point wrapper.
    `config_path` is accepted for future use but ignored for now –
    the module still relies on config.py constants.
    """
    all_results = []
    combos = [(d, 1.0, 0.0, "SCV") for d in DEMAND_FILES] + [
        (d, a, c, "MCV") for d in DEMAND_FILES for a in ALPHA_GRID for c in C_VALUES
    ]
    for demand_path, alpha, c, fleet_type in tqdm(combos):
        json_path = (
            RESULTS_RAW / f"{demand_path.stem}_{fleet_type}_{alpha:.2f}_{c:.0f}.json"
        )
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            # Ensure loaded data also has consistent types
            data = convert_numpy_types(data)
        else:
            if fleet_type == "SCV":
                params = make_scv_fleet(demand_path.stem)
            else:
                params = make_mcv_fleet(alpha, c, demand_path.stem)
            data = run_day(demand_path, params, fleet_type, alpha, c)
            # Convert numpy types to native Python types for JSON serialization
            data = convert_numpy_types(data)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        all_results.append(data)
    df = pd.DataFrame(all_results)
    df.to_parquet(SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
