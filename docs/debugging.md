# MILP Model Debugging

FleetMix provides a comprehensive debugging system for Mixed Integer Linear Programming (MILP) models that works seamlessly across different solvers (CBC, Gurobi, CPLEX).

## Overview

The MILP debugging feature allows you to export model artifacts for detailed analysis without modifying your optimization code. This is particularly useful for:

- Diagnosing infeasible models
- Understanding model structure
- Sharing models with collaborators
- Debugging solver-specific issues

## Usage

### Command Line Interface

Enable MILP debugging by adding the `--debug-milp` flag to any optimization command:

```bash
# Basic usage
fleetmix optimize --demand demand.csv --config config.yaml --debug-milp debug_output/

# With benchmarks
fleetmix benchmark mcvrp --instance 10_3_3_1_(01) --debug-milp debug_output/
```

### Programmatic Usage

You can also enable debugging programmatically:

```python
from fleetmix.utils.debug import ModelDebugger

# Enable debugging before optimization
ModelDebugger.enable(
    debug_dir="debug_output",
    artifacts={"lp", "mps", "solver_log", "iis"}  # Optional: specify which artifacts to generate
)

# Run your optimization as normal
solution = optimize(...)
```

## Generated Artifacts

The debugger can generate the following artifacts:

| Artifact | Extension | Description | Solver Support |
|----------|-----------|-------------|----------------|
| LP file | `.lp` | Human-readable linear programming format | All solvers |
| MPS file | `.mps` | Mathematical Programming System format | All solvers |
| Solver log | `.log` | Detailed solver output and progress | All solvers |
| IIS file | `.iis` | Irreducible Infeasible Set (for infeasible models) | Gurobi only |

### LP File Format

The LP file contains the complete model in a readable format:

```
\* Model *\
Minimize
Total_Cost: 73.333 x_4_1 + 73.333 x_5_1 + ...

Subject To
Customer_Coverage_C001: x_4_1 + x_5_1 + x_6_1 + x_7_1 = 1
Vehicle_Assignment_1: x_4_1 + x_5_1 + x_6_1 + x_7_1 - y_1 = 0
...

Binaries
x_4_1
x_5_1
...
```

### Solver Log

The solver log captures the complete output from the solver, including:
- Preprocessing statistics
- Solution progress
- Cut generation
- Branching decisions
- Final solution status

### IIS (Infeasible Models)

When a model is infeasible and you're using Gurobi, the debugger will attempt to compute an Irreducible Infeasible Set (IIS) - a minimal set of constraints that cause the infeasibility.

## Best Practices

1. **Use descriptive directory names**: Include timestamp or problem instance in the debug directory name
   ```bash
   fleetmix optimize --demand demand.csv --debug-milp debug_2024_01_15_instance_50/
   ```

2. **Clean up debug files**: Debug files can be large, especially for big instances. Remove them after analysis.

3. **Selective artifact generation**: If you only need specific artifacts, configure them explicitly:
   ```python
   ModelDebugger.enable(debug_dir="debug/", artifacts={"lp"})  # Only generate LP files
   ```

4. **Infeasibility debugging workflow**:
   - First, check the LP file to understand the model structure
   - Review the solver log for preprocessing issues
   - If using Gurobi, examine the IIS file to identify conflicting constraints

## Solver-Agnostic Design

The debugging system is designed to work with any solver supported by FleetMix:

- **CBC (default)**: Generates LP, MPS, and solver logs
- **Gurobi**: All CBC features plus IIS computation for infeasible models
- **CPLEX**: Same as CBC (IIS support could be added if needed)

The system gracefully handles solver-specific features - if a solver doesn't support a particular artifact type, it's silently skipped.

## Example: Debugging an Infeasible Model

```bash
# Run optimization with debugging enabled
fleetmix optimize --demand problematic_demand.csv --config config.yaml --debug-milp debug/

# If the model is infeasible, check the generated files:
ls debug/
# fsm_model.lp    # Model structure
# fsm_model.mps   # Alternative format
# fsm_model.log   # Solver output showing infeasibility
# fsm_model.iis   # (Gurobi only) Minimal infeasible constraint set

# Examine the IIS to understand the conflict
cat debug/fsm_model.iis
```

## Troubleshooting

**Q: The solver log is empty**
A: This can happen if the model solves very quickly. The log is generated by re-solving the model with verbose output enabled.

**Q: No IIS file is generated**
A: IIS computation is only available with Gurobi and only for infeasible models. Check that you're using Gurobi (`FSM_SOLVER=gurobi`) and that the model is actually infeasible.

**Q: MPS file generation fails**
A: Some model types may not support MPS format. The LP file is always generated as a fallback. 