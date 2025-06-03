"""
Command-line interface for Fleetmix using Typer.
"""
import sys
from pathlib import Path
from typing import Optional
import time

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from fleetmix import __version__
from fleetmix.api import optimize as api_optimize
from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.utils.logging import LogLevel, setup_logging, log_progress, log_success, log_error
from fleetmix.core_types import FleetmixSolution
from fleetmix.config.parameters import Parameters

app = typer.Typer(
    help="Fleetmix: Fleet Size and Mix optimizer for heterogeneous fleets",
    add_completion=False,
)
console = Console()


def _get_default_config() -> Parameters:
    """Load default configuration to get default values for CLI parameters."""
    try:
        return Parameters.from_yaml()
    except Exception:
        # Fallback to hardcoded defaults if config loading fails
        return None


# Load default config once at module level
_DEFAULT_CONFIG = _get_default_config()


def _get_available_instances(suite: str) -> list[str]:
    """Get list of available instances for a benchmark suite."""
    datasets_dir = Path(__file__).parent / "benchmarking" / "datasets"
    
    if suite == "mcvrp":
        mcvrp_dir = datasets_dir / "mcvrp"
        instances = [f.stem for f in sorted(mcvrp_dir.glob("*.dat"))]
    elif suite == "cvrp":
        cvrp_dir = datasets_dir / "cvrp"
        instances = [f.stem for f in sorted(cvrp_dir.glob("X-n*.vrp"))]
    else:
        instances = []
    
    return instances


def _list_instances(suite: str) -> None:
    """Display available instances for a benchmark suite."""
    instances = _get_available_instances(suite)
    
    if not instances:
        console.print(f"[yellow]No instances found for {suite.upper()}[/yellow]")
        return
    
    table = Table(title=f"Available {suite.upper()} Instances", show_header=True)
    table.add_column("Instance", style="cyan")
    
    for instance in instances:
        table.add_row(instance)
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(instances)} instances[/dim]")
    console.print(f"[dim]Usage: fleetmix benchmark {suite} --instance INSTANCE_NAME[/dim]")


def _run_single_instance(suite: str, instance: str, output_dir: Optional[Path] = None, format: str = "json", verbose: bool = False) -> None:
    """Run a single benchmark instance."""
    if suite == "mcvrp":
        # Run single MCVRP instance
        datasets_dir = Path(__file__).parent / "benchmarking" / "datasets" / "mcvrp"
        dat_path = datasets_dir / f"{instance}.dat"
        
        if not dat_path.exists():
            log_error(f"MCVRP instance '{instance}' not found")
            available = _get_available_instances("mcvrp")
            console.print(f"[yellow]Available instances:[/yellow] {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")
            console.print(f"[dim]Use 'fleetmix benchmark mcvrp --list' to see all available instances[/dim]")
            raise typer.Exit(1)
        
        log_progress(f"Running MCVRP instance {instance}...")
        
        # Use the unified pipeline interface for conversion
        customers_df, params = convert_to_fsm(
            VRPType.MCVRP,
            instance_path=dat_path
        )
        
        # Override output directory if specified
        if output_dir:
            params.results_dir = output_dir
        
        # Use the unified pipeline interface for optimization
        start_time = time.time()
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=verbose
        )

        # Save results with specified format
        ext = "xlsx" if format == "excel" else "json"
        output_path = params.results_dir / f"mcvrp_{instance}.{ext}"
        save_optimization_results(
            solution=solution,
            configurations_df=configs_df,
            parameters=params,
            filename=str(output_path),
            format=format,
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles
        )
        
        # Display results summary table
        table = Table(title=f"MCVRP Benchmark Results: {instance}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Cost", f"${solution.total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution.total_fixed_cost:,.2f}")
        table.add_row("Variable Cost", f"${solution.total_variable_cost:,.2f}")
        table.add_row("Penalties", f"${solution.total_penalties:,.2f}")
        table.add_row("Vehicles Used", str(solution.vehicles_used))
        table.add_row("Expected Vehicles", str(params.expected_vehicles))
        table.add_row("Missing Customers", str(len(solution.missing_customers)))
        table.add_row("Solver Status", solution.solver_status)
        table.add_row("Solver Time", f"{solution.solver_runtime_sec:.1f}s")
        
        # Add cluster load percentages if available
        if not solution.selected_clusters.empty:
            for i, (_, cluster) in enumerate(solution.selected_clusters.iterrows()):
                # Try to get load percentage from different possible columns
                load_pct = None
                if 'Load_total_pct' in cluster:
                    load_pct = cluster['Load_total_pct'] * 100  # Convert to percentage
                elif 'Vehicle_Utilization' in cluster:
                    load_pct = cluster['Vehicle_Utilization'] * 100  # Convert to percentage
                elif 'Total_Demand' in cluster and 'Config_ID' in cluster:
                    # Calculate load percentage from total demand and vehicle capacity
                    config = configs_df[configs_df['Config_ID'] == cluster['Config_ID']].iloc[0]
                    if isinstance(cluster['Total_Demand'], dict):
                        total_demand = sum(cluster['Total_Demand'].values())
                    elif isinstance(cluster['Total_Demand'], str):
                        import ast
                        total_demand = sum(ast.literal_eval(cluster['Total_Demand']).values())
                    else:
                        total_demand = cluster['Total_Demand']
                    load_pct = (total_demand / config['Capacity']) * 100
                
                if load_pct is not None:
                    table.add_row(f"Cluster {cluster['Cluster_ID']} Load %", f"{load_pct:.1f}%")
        
        console.print(table)
        log_success(f"Results saved to {output_path.name}")
        
    elif suite == "cvrp":
        # Run single CVRP instance
        available = _get_available_instances("cvrp")
        if instance not in available:
            log_error(f"CVRP instance '{instance}' not found")
            console.print(f"[yellow]Available instances:[/yellow] {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")
            console.print(f"[dim]Use 'fleetmix benchmark cvrp --list' to see all available instances[/dim]")
            raise typer.Exit(1)
        
        log_progress(f"Running CVRP instance {instance}...")
        
        # Use the unified pipeline interface for conversion
        # CVRP requires benchmark_type and uses instance_names instead of instance_path
        customers_df, params = convert_to_fsm(
            VRPType.CVRP,
            instance_names=[instance],
            benchmark_type=CVRPBenchmarkType.NORMAL
        )
        
        # Override output directory if specified
        if output_dir:
            params.results_dir = output_dir
        
        # Use the unified pipeline interface for optimization
        start_time = time.time()
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=verbose
        )

        # Save results with specified format
        ext = "xlsx" if format == "excel" else "json"
        output_path = params.results_dir / f"cvrp_{instance}_normal.{ext}"
        save_optimization_results(
            solution=solution,
            configurations_df=configs_df,
            parameters=params,
            filename=str(output_path),
            format=format,
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles
        )
        
        # Display results summary table
        table = Table(title=f"CVRP Benchmark Results: {instance}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Cost", f"${solution.total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution.total_fixed_cost:,.2f}")
        table.add_row("Variable Cost", f"${solution.total_variable_cost:,.2f}")
        table.add_row("Penalties", f"${solution.total_penalties:,.2f}")
        table.add_row("Vehicles Used", str(solution.vehicles_used))
        table.add_row("Expected Vehicles", str(params.expected_vehicles))
        table.add_row("Missing Customers", str(len(solution.missing_customers)))
        table.add_row("Solver Status", solution.solver_status)
        table.add_row("Solver Time", f"{solution.solver_runtime_sec:.1f}s")
        
        # Add cluster load percentages if available
        if not solution.selected_clusters.empty:
            for i, (_, cluster) in enumerate(solution.selected_clusters.iterrows()):
                # Try to get load percentage from different possible columns
                load_pct = None
                if 'Load_total_pct' in cluster:
                    load_pct = cluster['Load_total_pct'] * 100  # Convert to percentage
                elif 'Vehicle_Utilization' in cluster:
                    load_pct = cluster['Vehicle_Utilization'] * 100  # Convert to percentage
                elif 'Total_Demand' in cluster and 'Config_ID' in cluster:
                    # Calculate load percentage from total demand and vehicle capacity
                    config = configs_df[configs_df['Config_ID'] == cluster['Config_ID']].iloc[0]
                    if isinstance(cluster['Total_Demand'], dict):
                        total_demand = sum(cluster['Total_Demand'].values())
                    elif isinstance(cluster['Total_Demand'], str):
                        import ast
                        total_demand = sum(ast.literal_eval(cluster['Total_Demand']).values())
                    else:
                        total_demand = cluster['Total_Demand']
                    load_pct = (total_demand / config['Capacity']) * 100
                
                if load_pct is not None:
                    table.add_row(f"Cluster {cluster['Cluster_ID']} Load %", f"{load_pct:.1f}%")
        
        console.print(table)
        log_success(f"Results saved to {output_path.name}")


def _run_all_mcvrp_instances(output_dir: Optional[Path] = None, format: str = "json", verbose: bool = False, debug: bool = False) -> None:
    """Run benchmarks for all MCVRP instances."""
    datasets_dir = Path(__file__).parent / "benchmarking" / "datasets" / "mcvrp"
    
    for dat_path in sorted(datasets_dir.glob("*.dat")):
        instance = dat_path.stem
        log_progress(f"Running MCVRP instance {instance}...")
        
        try:
            # Use the unified pipeline interface for conversion
            customers_df, params = convert_to_fsm(
                VRPType.MCVRP,
                instance_path=dat_path
            )
            
            # Override output directory if specified
            if output_dir:
                params.results_dir = output_dir
            
            # Use the unified pipeline interface for optimization
            start_time = time.time()
            solution, configs_df = run_optimization(
                customers_df=customers_df,
                params=params,
                verbose=verbose
            )

            # Save results with specified format
            ext = "xlsx" if format == "excel" else "json"
            output_path = params.results_dir / f"mcvrp_{instance}.{ext}"
            save_optimization_results(
                solution=solution,
                configurations_df=configs_df,
                parameters=params,
                filename=str(output_path),
                format=format,
                is_benchmark=True,
                expected_vehicles=params.expected_vehicles
            )
            log_success(f"Saved results to {output_path.name}")
            
        except Exception as e:
            log_error(f"Error processing MCVRP instance {instance}: {e}")
            if debug:
                console.print_exception()


def _run_all_cvrp_instances(output_dir: Optional[Path] = None, format: str = "json", verbose: bool = False, debug: bool = False) -> None:
    """Run benchmarks for all CVRP instances."""
    datasets_dir = Path(__file__).parent / "benchmarking" / "datasets" / "cvrp"
    
    for vrp_path in sorted(datasets_dir.glob("X-n*.vrp")):
        instance = vrp_path.stem
        log_progress(f"Running CVRP instance {instance}...")
        
        try:
            # Use the unified pipeline interface for conversion
            # CVRP requires benchmark_type and uses instance_names instead of instance_path
            customers_df, params = convert_to_fsm(
                VRPType.CVRP,
                instance_names=[instance],
                benchmark_type=CVRPBenchmarkType.NORMAL
            )
            
            # Override output directory if specified
            if output_dir:
                params.results_dir = output_dir
            
            # Use the unified pipeline interface for optimization
            start_time = time.time()
            solution, configs_df = run_optimization(
                customers_df=customers_df,
                params=params,
                verbose=verbose
            )

            # Save results with specified format
            ext = "xlsx" if format == "excel" else "json"
            output_path = params.results_dir / f"cvrp_{instance}_normal.{ext}"
            save_optimization_results(
                solution=solution,
                configurations_df=configs_df,
                parameters=params,
                filename=str(output_path),
                format=format,
                is_benchmark=True,
                expected_vehicles=params.expected_vehicles
            )
            log_success(f"Saved results to {output_path.name}")
            
        except Exception as e:
            log_error(f"Error processing CVRP instance {instance}: {e}")
            if debug:
                console.print_exception()


@app.command()
def optimize(
    demand: Path = typer.Option(..., "--demand", "-d", help="Path to customer demand CSV file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration YAML file"),
    output: Path = typer.Option("results", "--output", "-o", help="Output directory"),
    format: str = typer.Option(
        _DEFAULT_CONFIG.format if _DEFAULT_CONFIG else "excel", 
        "--format", "-f", 
        help="Output format (excel or json)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (errors only)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
) -> None:
    """
    Optimize fleet size and mix for given customer demand.
    
    This command loads customer demand data, generates vehicle configurations,
    creates clusters, and solves the optimization problem to find the best
    fleet composition and routing solution.
    """
    # Setup logging based on flags
    _setup_logging_from_flags(verbose, quiet, debug)
    
    # Validate inputs
    if not demand.exists():
        log_error(f"Demand file not found: {demand}")
        raise typer.Exit(1)
        
    if config and not config.exists():
        log_error(f"Config file not found: {config}")
        raise typer.Exit(1)
        
    if format not in ["excel", "json"]:
        log_error("Invalid format. Choose 'excel' or 'json'")
        raise typer.Exit(1)
    
    try:
        # Show progress only for normal and verbose levels
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running optimization...", total=None)
                
                # Call the API
                solution = api_optimize(
                    demand=str(demand),
                    config=str(config) if config else None,
                    output_dir=str(output),
                    format=format,
                    verbose=verbose
                )
                
                progress.update(task, completed=True)
        else:
            # Run without progress spinner in quiet mode
            solution = api_optimize(
                demand=str(demand),
                config=str(config) if config else None,
                output_dir=str(output),
                format=format,
                verbose=verbose
            )
        
        # Display results summary (always shown unless quiet)
        table = Table(title="Optimization Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Cost", f"${solution.total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution.total_fixed_cost:,.2f}")
        table.add_row("Variable Cost", f"${solution.total_variable_cost:,.2f}")
        table.add_row("Penalties", f"${solution.total_penalties:,.2f}")
        table.add_row("Vehicles Used", str(solution.total_vehicles))
        table.add_row("Missing Customers", str(len(solution.missing_customers)))
        table.add_row("Solver Status", solution.solver_status)
        table.add_row("Solver Time", f"{solution.solver_runtime_sec:.1f}s")
        
        console.print(table)
        log_success(f"Results saved to {output}/")
        
    except FileNotFoundError as e:
        log_error(str(e))
        raise typer.Exit(1)
    except ValueError as e:
        log_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def benchmark(
    suite: str = typer.Argument(..., help="Benchmark suite to run: 'mcvrp' or 'cvrp'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    instance: Optional[str] = typer.Option(None, "--instance", "-i", help="Specific instance to run (if not specified, runs all instances)"),
    format: str = typer.Option(
        _DEFAULT_CONFIG.format if _DEFAULT_CONFIG else "json", 
        "--format", "-f", 
        help="Output format (excel or json)"
    ),
    list_instances: bool = typer.Option(False, "--list", "-l", help="List available instances and exit"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (errors only)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
) -> None:
    """
    Run benchmark suite on standard VRP instances.
    
    By default runs all instances in the suite. Use --instance to run a specific instance.
    Use --list to see all available instances for the suite.
    """
    # Setup logging based on flags
    _setup_logging_from_flags(verbose, quiet, debug)
    
    if suite not in ["mcvrp", "cvrp"]:
        log_error(f"Invalid suite '{suite}'. Choose 'mcvrp' or 'cvrp'")
        raise typer.Exit(1)
    
    if format not in ["excel", "json"]:
        log_error("Invalid format. Choose 'excel' or 'json'")
        raise typer.Exit(1)
    
    # Handle --list flag
    if list_instances:
        _list_instances(suite)
        return
    
    if instance:
        # Run single instance
        _run_single_instance(suite, instance, output, format, verbose)
    else:
        # Run all instances
        try:
            if not quiet:
                log_progress(f"Running {suite.upper()} benchmark suite...")
                
            if suite == "mcvrp":
                # Implement batch MCVRP processing using pipeline interface
                _run_all_mcvrp_instances(output, format, verbose, debug)
                        
            else:  # cvrp
                # Implement batch CVRP processing using pipeline interface
                _run_all_cvrp_instances(output, format, verbose, debug)
            
            log_success(f"{suite.upper()} benchmark completed successfully!")
                
        except Exception as e:
            log_error(f"Error running benchmark: {e}")
            if debug:
                console.print_exception()
            raise typer.Exit(1)


@app.command()
def convert(
    type: str = typer.Option(..., "--type", "-t", help="VRP type: 'cvrp' or 'mcvrp'"),
    instance: str = typer.Option(..., "--instance", "-i", help="Instance name"),
    benchmark_type: Optional[str] = typer.Option(None, "--benchmark-type", "-b", 
                                                  help="Benchmark type for CVRP: normal, split, scaled, combined"),
    num_goods: int = typer.Option(3, "--num-goods", help="Number of goods for CVRP (2 or 3)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option(
        _DEFAULT_CONFIG.format if _DEFAULT_CONFIG else "excel", 
        "--format", "-f", 
        help="Output format (excel or json)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (errors only)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
) -> None:
    """
    Convert VRP instances to FSM format and optimize.
    """
    # Setup logging based on flags
    _setup_logging_from_flags(verbose, quiet, debug)
    
    if type not in ["cvrp", "mcvrp"]:
        log_error(f"Invalid type '{type}'. Choose 'cvrp' or 'mcvrp'")
        raise typer.Exit(1)
        
    vrp_type = VRPType(type)
    
    # Validate CVRP-specific options
    if vrp_type == VRPType.CVRP:
        if not benchmark_type:
            log_error("--benchmark-type is required for CVRP")
            raise typer.Exit(1)
        if benchmark_type not in ["normal", "split", "scaled", "combined"]:
            log_error(f"Invalid benchmark type '{benchmark_type}'")
            raise typer.Exit(1)
        if num_goods not in [2, 3]:
            log_error("num_goods must be 2 or 3")
            raise typer.Exit(1)
            
    try:
        if not quiet:
            log_progress(f"Converting {type.upper()} instance '{instance}'...")
            
        if vrp_type == VRPType.CVRP:
            bench_type = CVRPBenchmarkType(benchmark_type)
            customers_df, params = convert_to_fsm(
                vrp_type,
                instance_names=[instance],
                benchmark_type=bench_type,
                num_goods=num_goods,
            )
            filename_stub = f"vrp_{type}_{instance}_{benchmark_type}"
        else:  # MCVRP
            instance_path = (
                Path(__file__).parent / "benchmarking" / "datasets" / "mcvrp" / f"{instance}.dat"
            )
            if not instance_path.exists():
                log_error(f"MCVRP instance file not found: {instance_path}")
                raise typer.Exit(1)
                
            customers_df, params = convert_to_fsm(
                vrp_type,
                instance_path=instance_path,
            )
            filename_stub = f"vrp_{type}_{instance}"
                
        # Override output directory if specified
        if output:
            params.results_dir = output
            
        # Run optimization
        if not quiet:
            log_progress("Running optimization on converted instance...")
        start_time = time.time()
        
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=verbose,
        )
        
        # Save results
        ext = "xlsx" if format == "excel" else "json"
        results_path = params.results_dir / f"{filename_stub}.{ext}"
        
        save_optimization_results(
            solution=solution,
            configurations_df=configs_df,
            parameters=params,
            filename=results_path,
            format=format,
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles,
        )
        
        log_success("Conversion and optimization completed!")
        log_success(f"Results saved to {results_path}")
        
    except FileNotFoundError as e:
        log_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        log_error(f"Error during conversion: {e}")
        if debug:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def gui(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run GUI on"),
) -> None:
    """
    Launch the web-based GUI for optimization.
    
    This starts a Streamlit web interface where you can:
    - Upload customer demand data
    - Configure optimization parameters
    - Monitor optimization progress
    - View and download results
    """
    console.print("[bold cyan]Launching Fleetmix GUI...[/bold cyan]")
    
    try:
        import streamlit
        import subprocess
        import sys
        from pathlib import Path
        
        gui_file = Path(__file__).parent / "gui.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(gui_file), "--server.port", str(port)]
        
        console.print(f"[green]✓[/green] GUI running at: http://localhost:{port}")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]")
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            console.print("\n[yellow]GUI server stopped[/yellow]")
            
    except ImportError:
        console.print("[red]Error: GUI dependencies not installed[/red]")
        console.print("Install with: pip install fleetmix")
        raise typer.Exit(1)

@app.command()
def version() -> None:
    """
    Show the Fleetmix version.
    """
    console.print(f"Fleetmix version {__version__}")


def _setup_logging_from_flags(verbose: bool = False, quiet: bool = False, debug: bool = False):
    """Setup logging based on CLI flags or environment variable."""
    level_from_flags: Optional[LogLevel] = None
    if debug:
        level_from_flags = LogLevel.DEBUG
    elif verbose:
        level_from_flags = LogLevel.VERBOSE
    elif quiet:
        level_from_flags = LogLevel.QUIET

    if level_from_flags is not None:
        setup_logging(level_from_flags)
    else:
        # No flags set, let setup_logging handle it (will check env var)
        setup_logging()


if __name__ == "__main__":
    app() 