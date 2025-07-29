"""
Generate SCV baselines for existing MCV results.
"""

import json
from pathlib import Path
from tqdm import tqdm
from experiments.alpha_analysis.config import DEMAND_FILES
from experiments.alpha_analysis.fleet_templates import make_scv_fleet
from experiments.alpha_analysis.run_day import run_day
from experiments.alpha_analysis.run_grid import convert_numpy_types

RESULTS_RAW = Path("results_raw")

def get_days_with_mcv_results():
    """Find which days already have MCV results."""
    mcv_files = list(RESULTS_RAW.glob("*_MCV_*.json"))
    days = set()
    for f in mcv_files:
        # Extract day_id from filename like "sales_2024-06-01_demand_MCV_1.00_0.json"
        day_id = f.stem.split('_MCV_')[0]
        days.add(day_id)
    return sorted(days)

def main():
    days_with_mcv = get_days_with_mcv_results()
    print(f"Found MCV results for {len(days_with_mcv)} days")
    
    if len(days_with_mcv) == 0:
        print("No MCV results found!")
        return
    
    scv_params = make_scv_fleet()
    completed = 0
    
    for day_id in tqdm(days_with_mcv, desc="Processing SCV baselines"):
        scv_path = RESULTS_RAW / f"{day_id}_SCV_1.00_0.json"
        if scv_path.exists():
            completed += 1
            continue
            
        # Find the corresponding demand file
        demand_file = None
        for d in DEMAND_FILES:
            if d.stem == day_id:
                demand_file = d
                break
        
        if demand_file is None:
            print(f"Warning: No demand file found for {day_id}")
            continue
            
        try:
            print(f"Processing SCV for {day_id}...")
            data = run_day(demand_file, scv_params, 'SCV', 1.0, 0.0)
            # Convert numpy types to ensure JSON serialization works
            data = convert_numpy_types(data)
            
            with open(scv_path, 'w') as f:
                json.dump(data, f, indent=2)
            completed += 1
            
        except Exception as e:
            print(f"Error processing {day_id}: {e}")
    
    print(f"\nCompleted SCV processing for {completed}/{len(days_with_mcv)} days")
    print(f"You can now run the partial analysis for meaningful MCV vs SCV comparison!")

if __name__ == "__main__":
    main() 