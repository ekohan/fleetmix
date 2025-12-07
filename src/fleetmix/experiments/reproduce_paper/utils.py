"""
Shared utilities for paper reproducibility runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

__all__ = [
    "ProgressTracker",
    "aggregate_results_to_dataframe",
    "ensure_output_dir",
    "load_json_results",
    "parse_comma_separated",
    "print_summary_stats",
    "save_summary_table",
    "skip_if_exists",
]

console = Console()
T = TypeVar("T")


class ProgressTracker:
    """Wrapper for rich progress bars with consistent styling."""

    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id: TaskID | None = None

    def __enter__(self):
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update(self, advance: int = 1, description: str | None = None):
        """Update progress bar."""
        if self.task_id is not None:
            if description:
                self.progress.update(
                    self.task_id, advance=advance, description=description
                )
            else:
                self.progress.update(self.task_id, advance=advance)

    def set_description(self, description: str):
        """Update the task description."""
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)


def skip_if_exists(output_path: Path) -> bool:
    """Check if output file already exists."""
    return output_path.exists() and output_path.stat().st_size > 0


def load_json_results(
    results_dir: Path, pattern: str = "*.json"
) -> list[dict[str, Any]]:
    """Load all JSON results from a directory."""
    results = []
    for json_path in results_dir.rglob(pattern):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            console.print(
                f"[yellow]Warning: Skipping corrupted file {json_path}: {e}[/yellow]"
            )
            continue
    return results


def aggregate_results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert list of result dictionaries to pandas DataFrame."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def save_summary_table(
    df: pd.DataFrame, output_path: Path, formats: list[str] = ["parquet", "csv"]
):
    """Save DataFrame in multiple formats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "parquet" in formats:
        parquet_path = output_path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        console.print(f"[green]✓[/green] Saved summary to {parquet_path}")

    if "csv" in formats:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        console.print(f"[green]✓[/green] Saved summary to {csv_path}")


def print_summary_stats(title: str, stats: dict[str, Any]):
    """Print summary statistics in a nice table."""
    table = Table(title=title, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        table.add_row(key, formatted_value)

    console.print(table)


def parse_comma_separated(
    value: str | None, valid_options: list[str] | None = None
) -> list[str]:
    """Parse comma-separated string into list, with validation."""
    if value is None or value.lower() == "all":
        return valid_options if valid_options else []

    items = [item.strip() for item in value.split(",")]

    if valid_options:
        invalid = [item for item in items if item not in valid_options]
        if invalid:
            raise ValueError(
                f"Invalid items: {invalid}. Valid options: {valid_options}"
            )

    return items


def ensure_output_dir(output_dir: Path, confirm_overwrite: bool = False) -> Path:
    """
    Ensure output directory exists, with optional overwrite confirmation.

    Args:
        output_dir: Target output directory
        confirm_overwrite: If True and dir has files, ask for confirmation

    Returns:
        The output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if confirm_overwrite and any(output_dir.iterdir()):
        console.print(
            f"[yellow]Warning: Output directory {output_dir} already contains files.[/yellow]"
        )
        response = console.input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            console.print("[red]Aborted.[/red]")
            raise SystemExit(1)

    return output_dir
