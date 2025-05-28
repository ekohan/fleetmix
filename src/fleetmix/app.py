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

app = typer.Typer(
    help="Fleetmix: Fleet Size and Mix optimizer for heterogeneous fleets",
    add_completion=False,
)
console = Console()


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


def _run_single_instance(suite: str, instance: str, output_dir: Optional[Path] = None, verbose: bool = False) -> None:
    """Run a single benchmark instance."""
    if suite == "mcvrp":
        # Run single MCVRP instance
        datasets_dir = Path(__file__).parent / "benchmarking" / "datasets" / "mcvrp"
        dat_path = datasets_dir / f"{instance}.dat"
        
        if not dat_path.exists():
            console.print(f"[red]Error:[/red] MCVRP instance '{instance}' not found")
            available = _get_available_instances("mcvrp")
            console.print(f"[yellow]Available instances:[/yellow] {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")
            console.print(f"[dim]Use 'fleetmix benchmark mcvrp --list' to see all available instances[/dim]")
            raise typer.Exit(1)
        
        console.print(f"Running MCVRP instance {instance}...")
        
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

        # Save results as JSON for later batch analysis
        output_path = params.results_dir / f"mcvrp_{instance}.json"
        save_optimization_results(
            execution_time=time.time() - start_time,
            solver_name=solution["solver_name"],
            solver_status=solution["solver_status"],
            solver_runtime_sec=solution["solver_runtime_sec"],
            post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
            configurations_df=configs_df,
            selected_clusters=solution["selected_clusters"],
            total_fixed_cost=solution["total_fixed_cost"],
            total_variable_cost=solution["total_variable_cost"],
            total_light_load_penalties=solution["total_light_load_penalties"],
            total_compartment_penalties=solution["total_compartment_penalties"],
            total_penalties=solution["total_penalties"],
            vehicles_used=solution["vehicles_used"],
            missing_customers=solution["missing_customers"],
            parameters=params,
            filename=str(output_path),
            format="json",
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles
        )
        
        # Display results summary table
        table = Table(title=f"MCVRP Benchmark Results: {instance}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        total_cost = (
            solution['total_fixed_cost'] + 
            solution['total_variable_cost'] + 
            solution['total_penalties']
        )
        
        table.add_row("Total Cost", f"${total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution['total_fixed_cost']:,.2f}")
        table.add_row("Variable Cost", f"${solution['total_variable_cost']:,.2f}")
        table.add_row("Penalties", f"${solution['total_penalties']:,.2f}")
        table.add_row("Vehicles Used", str(solution['vehicles_used']))
        table.add_row("Expected Vehicles", str(params.expected_vehicles))
        table.add_row("Missing Customers", str(len(solution['missing_customers'])))
        table.add_row("Solver Status", solution['solver_status'])
        table.add_row("Solver Time", f"{solution['solver_runtime_sec']:.1f}s")
        
        # Add cluster load percentages if available
        if 'selected_clusters' in solution and not solution['selected_clusters'].empty:
            for i, (_, cluster) in enumerate(solution['selected_clusters'].iterrows()):
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
        console.print(f"[green]✓[/green] Results saved to {output_path.name}")
        
    elif suite == "cvrp":
        # Run single CVRP instance
        available = _get_available_instances("cvrp")
        if instance not in available:
            console.print(f"[red]Error:[/red] CVRP instance '{instance}' not found")
            console.print(f"[yellow]Available instances:[/yellow] {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")
            console.print(f"[dim]Use 'fleetmix benchmark cvrp --list' to see all available instances[/dim]")
            raise typer.Exit(1)
        
        console.print(f"Running CVRP instance {instance}...")
        
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

        # Save results as JSON for later batch analysis
        output_path = params.results_dir / f"cvrp_{instance}_normal.json"
        save_optimization_results(
            execution_time=time.time() - start_time,
            solver_name=solution["solver_name"],
            solver_status=solution["solver_status"],
            solver_runtime_sec=solution["solver_runtime_sec"],
            post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
            configurations_df=configs_df,
            selected_clusters=solution["selected_clusters"],
            total_fixed_cost=solution["total_fixed_cost"],
            total_variable_cost=solution["total_variable_cost"],
            total_light_load_penalties=solution["total_light_load_penalties"],
            total_compartment_penalties=solution["total_compartment_penalties"],
            total_penalties=solution["total_penalties"],
            vehicles_used=solution["vehicles_used"],
            missing_customers=solution["missing_customers"],
            parameters=params,
            filename=str(output_path),
            format="json",
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles
        )
        
        # Display results summary table
        table = Table(title=f"CVRP Benchmark Results: {instance}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        total_cost = (
            solution['total_fixed_cost'] + 
            solution['total_variable_cost'] + 
            solution['total_penalties']
        )
        
        table.add_row("Total Cost", f"${total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution['total_fixed_cost']:,.2f}")
        table.add_row("Variable Cost", f"${solution['total_variable_cost']:,.2f}")
        table.add_row("Penalties", f"${solution['total_penalties']:,.2f}")
        table.add_row("Vehicles Used", str(solution['vehicles_used']))
        table.add_row("Expected Vehicles", str(params.expected_vehicles))
        table.add_row("Missing Customers", str(len(solution['missing_customers'])))
        table.add_row("Solver Status", solution['solver_status'])
        table.add_row("Solver Time", f"{solution['solver_runtime_sec']:.1f}s")
        
        # Add cluster load percentages if available
        if 'selected_clusters' in solution and not solution['selected_clusters'].empty:
            for i, (_, cluster) in enumerate(solution['selected_clusters'].iterrows()):
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
        console.print(f"[green]✓[/green] Results saved to {output_path.name}")


@app.command()
def optimize(
    demand: Path = typer.Option(..., "--demand", "-d", help="Path to customer demand CSV file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration YAML file"),
    output: Path = typer.Option("results", "--output", "-o", help="Output directory"),
    format: str = typer.Option("excel", "--format", "-f", help="Output format (excel or json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Run fleet optimization on customer demand data.
    """
    # Validate files exist
    if not demand.exists():
        console.print(f"[red]Error:[/red] Demand file not found: {demand}")
        raise typer.Exit(1)
        
    if config and not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)
        
    if format not in ["excel", "json"]:
        console.print(f"[red]Error:[/red] Invalid format. Choose 'excel' or 'json'")
        raise typer.Exit(1)
    
    try:
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
        
        # Display results summary
        table = Table(title="Optimization Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        total_cost = (
            solution['total_fixed_cost'] + 
            solution['total_variable_cost'] + 
            solution['total_penalties']
        )
        
        table.add_row("Total Cost", f"${total_cost:,.2f}")
        table.add_row("Fixed Cost", f"${solution['total_fixed_cost']:,.2f}")
        table.add_row("Variable Cost", f"${solution['total_variable_cost']:,.2f}")
        table.add_row("Penalties", f"${solution['total_penalties']:,.2f}")
        table.add_row("Vehicles Used", str(solution['total_vehicles']))
        table.add_row("Missing Customers", str(len(solution['missing_customers'])))
        table.add_row("Solver Status", solution['solver_status'])
        table.add_row("Solver Time", f"{solution['solver_runtime_sec']:.1f}s")
        
        console.print(table)
        console.print(f"\n[green]✓[/green] Results saved to {output}/")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def benchmark(
    suite: str = typer.Argument(..., help="Benchmark suite to run: 'mcvrp' or 'cvrp'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    instance: Optional[str] = typer.Option(None, "--instance", "-i", help="Specific instance to run (if not specified, runs all instances)"),
    list_instances: bool = typer.Option(False, "--list", "-l", help="List available instances and exit"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Run benchmark suite on standard VRP instances.
    
    By default runs all instances in the suite. Use --instance to run a specific instance.
    Use --list to see all available instances for the suite.
    """
    if suite not in ["mcvrp", "cvrp"]:
        console.print(f"[red]Error:[/red] Invalid suite '{suite}'. Choose 'mcvrp' or 'cvrp'")
        raise typer.Exit(1)
    
    # Handle --list flag
    if list_instances:
        _list_instances(suite)
        return
    
    if instance:
        # Run single instance
        _run_single_instance(suite, instance, output, verbose)
    else:
        # Run all instances
        try:
            with console.status(f"Running {suite.upper()} benchmark suite..."):
                if suite == "mcvrp":
                    from fleetmix.cli.run_all_mcvrp import main as run_mcvrp
                    # Temporarily redirect stdout to capture output
                    import io
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        run_mcvrp()
                        
                else:  # cvrp
                    from fleetmix.cli.run_all_cvrp import main as run_cvrp
                    f = io.StringIO()
                    with redirect_stdout(f):
                        run_cvrp()
            
            console.print(f"[green]✓[/green] {suite.upper()} benchmark completed successfully!")
            
            if verbose:
                console.print("\n[dim]Benchmark output:[/dim]")
                console.print(f.getvalue())
                
        except Exception as e:
            console.print(f"[red]Error running benchmark:[/red] {e}")
            if verbose:
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
    format: str = typer.Option("excel", "--format", "-f", help="Output format (excel or json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Convert VRP instances to FSM format and optimize.
    """
    if type not in ["cvrp", "mcvrp"]:
        console.print(f"[red]Error:[/red] Invalid type '{type}'. Choose 'cvrp' or 'mcvrp'")
        raise typer.Exit(1)
        
    vrp_type = VRPType(type)
    
    # Validate CVRP-specific options
    if vrp_type == VRPType.CVRP:
        if not benchmark_type:
            console.print("[red]Error:[/red] --benchmark-type is required for CVRP")
            raise typer.Exit(1)
        if benchmark_type not in ["normal", "split", "scaled", "combined"]:
            console.print(f"[red]Error:[/red] Invalid benchmark type '{benchmark_type}'")
            raise typer.Exit(1)
        if num_goods not in [2, 3]:
            console.print(f"[red]Error:[/red] num_goods must be 2 or 3")
            raise typer.Exit(1)
            
    try:
        with console.status(f"Converting {type.upper()} instance '{instance}'..."):
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
                    console.print(f"[red]Error:[/red] MCVRP instance file not found: {instance_path}")
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
        console.print(f"Running optimization on converted instance...")
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
            execution_time=time.time() - start_time,
            solver_name=solution["solver_name"],
            solver_status=solution["solver_status"],
            solver_runtime_sec=solution["solver_runtime_sec"],
            post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
            configurations_df=configs_df,
            selected_clusters=solution["selected_clusters"],
            total_fixed_cost=solution["total_fixed_cost"],
            total_variable_cost=solution["total_variable_cost"],
            total_light_load_penalties=solution["total_light_load_penalties"],
            total_compartment_penalties=solution["total_compartment_penalties"],
            total_penalties=solution["total_penalties"],
            vehicles_used=solution["vehicles_used"],
            missing_customers=solution["missing_customers"],
            parameters=params,
            filename=results_path,
            format=format,
            is_benchmark=True,
            expected_vehicles=params.expected_vehicles,
        )
        
        console.print(f"[green]✓[/green] Conversion and optimization completed!")
        console.print(f"[green]✓[/green] Results saved to {results_path}")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during conversion:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """
    Show the Fleetmix version.
    """
    console.print(f"Fleetmix version {__version__}")


if __name__ == "__main__":
    app() 