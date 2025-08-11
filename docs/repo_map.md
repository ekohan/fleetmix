
# FleetMix Repository Map

## Top-Level Directories
Based on the workspace snapshot and list_dir:

- .git/ (git repository)
- tests/ (unit and integration tests)
- README.md (project documentation)
- .coverage (coverage data)
- .hypothesis/ (hypothesis testing cache)
- FSM-MCV_Model2-pulp.sol (solver output)
- gurobi.log (solver log)
- FSM-MCV_Model2-pulp.lp (solver model)
- results/ (output results)
- .DS_Store (macOS file)
- repomix-output.xml (possibly generated file)
- Core model excerpt.tex (LaTeX excerpt)
- demand_scenarios.txt (demand info)
- paper_excerpts.txt (paper excerpts)
- summary_alpha.md (alpha analysis summary)
- Screenshot 2025-07-21 at 12.33.20 PM.png (screenshot)
- paper/ (paper-related files)
- tools/ (utility scripts)
- Screenshot 2025-07-19 at 2.17.18 PM.png (screenshot)
- Notes_17_July_2025.md (notes)
- Screenshot 2025-07-19 at 2.12.27 PM.png (screenshot)
- Screenshot 2025-07-19 at 2.11.55 PM.png (screenshot)
- .cursorignore (ignore file)
- .gitignore (git ignore)
- .mypy_cache/ (mypy cache)
- examples/ (example scripts)
- docs/ (documentation)
- data/ (data files and queries)
- diff.txt (diff file)
- .cursor/ (cursor-related)
- init.sh (init script)
- src/ (source code)
- .ruff_cache/ (ruff cache)
- coverage/ (coverage reports)
- .pytest_cache/ (pytest cache)
- fleetmix-env/ (virtual environment)
- results_xlsx.png (image)
- results_cluster_viz.png (image)
- pyproject.toml (project config)
- pytest.ini (pytest config)
- fsm.png (image)
- dist/ (distribution)
- LICENSE (license file)
- codecov.yml (codecov config)
- .github/ (GitHub workflows)

Note: No 'experiments/' directory exists yet; it may be created in later stages.

## Key Modules
- src/fleetmix/: Main package with submodules like benchmarking, clustering, config, core_types.py, gui.py, merging, optimization, pipeline, post_optimization, preprocess, registry.py, utils/
- tests/: Testing suite with unit, integration, etc.
- tools/: Scripts like alpha_analysis.py, analyze_exports.py, etc.
- data/: Demand profiles, import/export scripts, queries
- docs/: Documentation files like quickstart.md
- examples/: Custom examples for clustering, solvers, etc.

## Fixed Costs Location
Vehicle fixed costs are defined in YAML configuration files under src/fleetmix/config/, such as default_config.yaml, where each vehicle type has a 'fixed_cost' field.

Recommended import: from fleetmix.config.loader import load_fleetmix_params
(or directly access params in fleetmix.config.params.FleetmixParams after loading)

## Demand Loaders Location
Demand data is loaded from CSV files primarily in src/fleetmix/utils/data_processing.py via the load_customer_demand function, which handles paths, reading, pivoting, and formatting.

Recommended import: from fleetmix.utils.data_processing import load_customer_demand

## Dependency Check
Attempted 'poetry show' failed (command not found). The project uses pyproject.toml, likely managed by Poetry or similar, but may require virtualenv activation (see fleetmix-env/). Alternatively, dependencies can be inferred from imports in source files.

For full dependencies, run in activated env: pip list or poetry show if installed. 