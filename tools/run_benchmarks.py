#!/usr/bin/env python3
"""
Simple runner script for FleetMix benchmarking.

Runs MCVRP benchmarks twice, then case benchmarks for all config files.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print("✓ Command completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}\n")
        return False


def main():
    """Main runner function."""
    print("FleetMix Benchmark Runner")
    print("=" * 50)

    # Get the project root (parent of tools directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Change to project root directory
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}\n")

    # Run CVRP and MCVRP benchmarks
    print("Step 1: Running CVRP and MCVRP benchmarks")
    print("-" * 40)

    cvrp_cmd = (
        "fleetmix benchmark cvrp --config src/fleetmix/config/default_config.yaml"
    )
    mcvrp_cmd = (
        "fleetmix benchmark mcvrp --config src/fleetmix/config/default_config.yaml"
    )

    print("CVRP Benchmark:")
    if not run_command(cvrp_cmd):
        print("CVRP benchmark failed. Stopping execution.")
        sys.exit(1)

    print("MCVRP Benchmark:")
    if not run_command(mcvrp_cmd):
        print("MCVRP benchmark failed. Stopping execution.")
        sys.exit(1)

    # Get all config files
    print("Step 2: Running case benchmarks for all configs")
    print("-" * 40)

    config_pattern = "src/fleetmix/config/*.yaml"
    config_files = glob.glob(config_pattern)
    config_files.sort()  # Sort for consistent ordering

    if not config_files:
        print(f"No config files found matching: {config_pattern}")
        sys.exit(1)

    print(f"Found {len(config_files)} config files:")
    for config in config_files:
        print(f"  - {config}")
    print()

    # Run case benchmarks for each config
    failed_configs = []
    for i, config in enumerate(config_files, 1):
        print(f"Case Benchmark {i}/{len(config_files)} - Config: {config}")
        case_cmd = f"fleetmix benchmark case --config {config}"

        if not run_command(case_cmd):
            failed_configs.append(config)

    # Summary
    print("Benchmark Runner Summary")
    print("=" * 50)
    print(f"Total configs processed: {len(config_files)}")

    if failed_configs:
        print(f"Failed configs ({len(failed_configs)}):")
        for config in failed_configs:
            print(f"  - {config}")
        sys.exit(1)
    else:
        print("✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
