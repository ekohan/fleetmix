#!/usr/bin/env python3
"""
Script to analyze all exported interfaces in the FleetMix package.
This helps identify potential interface leakages and ensures minimal public API surface.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def parse_init_file(init_path: Path) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Parse an __init__.py file to extract:
    - __all__ exports (if defined)
    - import statements
    - from imports
    """
    if not init_path.exists():
        return [], set(), set()

    try:
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        # Extract __all__ if present
        all_exports = []
        imports = set()
        from_imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    all_exports.append(elt.value)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        from_imports.add(f"{module}.*")
                    else:
                        from_imports.add(f"{module}.{alias.name}")

        return all_exports, imports, from_imports

    except Exception as e:
        print(f"Error parsing {init_path}: {e}")
        return [], set(), set()


def get_module_attributes(module_path: Path) -> Set[str]:
    """Get all public attributes from a Python module (not starting with _)."""
    if not module_path.exists():
        return set()

    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        attributes = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                attributes.add(node.name)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                attributes.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        attributes.add(target.id)

        return attributes

    except Exception as e:
        print(f"Error parsing {module_path}: {e}")
        return set()


def analyze_package_exports(package_root: Path) -> Dict:
    """Analyze all exports in the FleetMix package."""
    src_root = package_root / "src" / "fleetmix"

    if not src_root.exists():
        raise RuntimeError(f"Package source not found at {src_root}")

    analysis = {"top_level": {}, "submodules": {}, "potential_leaks": [], "summary": {}}

    # Analyze top-level package
    main_init = src_root / "__init__.py"
    all_exports, imports, from_imports = parse_init_file(main_init)

    analysis["top_level"] = {
        "explicit_exports": all_exports,
        "imports": list(imports),
        "from_imports": list(from_imports),
        "path": str(main_init.relative_to(package_root)),
    }

    # Analyze all submodules
    for init_file in src_root.rglob("__init__.py"):
        if init_file == main_init:
            continue

        rel_path = init_file.relative_to(src_root)
        module_name = str(rel_path.parent).replace(os.sep, ".")

        all_exports, imports, from_imports = parse_init_file(init_file)

        analysis["submodules"][module_name] = {
            "explicit_exports": all_exports,
            "imports": list(imports),
            "from_imports": list(from_imports),
            "path": str(init_file.relative_to(package_root)),
        }

    # Check for potential leaks
    # 1. Modules with no __all__ defined
    for module_name, info in analysis["submodules"].items():
        if not info["explicit_exports"]:
            # Check if module has any Python files that could leak attributes
            module_dir = src_root / module_name.replace(".", os.sep)
            if module_dir.is_dir():
                py_files = list(module_dir.glob("*.py"))
                py_files = [f for f in py_files if f.name != "__init__.py"]
                if py_files:
                    analysis["potential_leaks"].append(
                        {
                            "type": "no_all_defined",
                            "module": module_name,
                            "reason": f"No __all__ defined but has {len(py_files)} Python files",
                            "files": [
                                str(f.relative_to(package_root)) for f in py_files
                            ],
                        }
                    )

    # 2. Modules exposed at top level that have many exports
    top_level_modules = []
    for export in analysis["top_level"]["explicit_exports"]:
        if export in analysis["submodules"]:
            top_level_modules.append(export)

    for module_name in top_level_modules:
        module_info = analysis["submodules"][module_name]
        if len(module_info["explicit_exports"]) > 10:
            analysis["potential_leaks"].append(
                {
                    "type": "large_exposed_module",
                    "module": module_name,
                    "reason": f"Module exposed at top level with {len(module_info['explicit_exports'])} exports",
                    "exports": module_info["explicit_exports"],
                }
            )

    # Generate summary
    total_top_level_exports = len(analysis["top_level"]["explicit_exports"])
    total_submodule_exports = sum(
        len(info["explicit_exports"]) for info in analysis["submodules"].values()
    )
    modules_with_all = sum(
        1 for info in analysis["submodules"].values() if info["explicit_exports"]
    )
    total_submodules = len(analysis["submodules"])

    analysis["summary"] = {
        "top_level_exports": total_top_level_exports,
        "total_submodule_exports": total_submodule_exports,
        "submodules_with_explicit_all": modules_with_all,
        "total_submodules": total_submodules,
        "potential_leaks_count": len(analysis["potential_leaks"]),
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print the analysis results in a readable format."""

    print("=" * 80)
    print("FLEETMIX PACKAGE INTERFACE ANALYSIS")
    print("=" * 80)

    # Summary
    summary = analysis["summary"]
    print("\nSUMMARY:")
    print(f"  Top-level exports: {summary['top_level_exports']}")
    print(f"  Total submodule exports: {summary['total_submodule_exports']}")
    print(
        f"  Submodules with explicit __all__: {summary['submodules_with_explicit_all']}/{summary['total_submodules']}"
    )
    print(f"  Potential leaks detected: {summary['potential_leaks_count']}")

    # Top-level exports
    print(
        f"\nTOP-LEVEL EXPORTS ({len(analysis['top_level']['explicit_exports'])} items):"
    )
    for export in sorted(analysis["top_level"]["explicit_exports"]):
        print(f"  • {export}")

    # Submodule breakdown
    print("\nSUBMODULE BREAKDOWN:")
    for module_name, info in sorted(analysis["submodules"].items()):
        export_count = len(info["explicit_exports"])
        status = "✓" if export_count > 0 else "⚠"
        print(f"  {status} {module_name}: {export_count} exports")
        if export_count > 0 and export_count <= 5:
            for export in info["explicit_exports"]:
                print(f"    - {export}")
        elif export_count > 5:
            print(
                f"    - {', '.join(info['explicit_exports'][:3])}... (+{export_count - 3} more)"
            )

    # Potential leaks
    if analysis["potential_leaks"]:
        print("\nPOTENTIAL INTERFACE LEAKS:")
        for i, leak in enumerate(analysis["potential_leaks"], 1):
            print(f"  {i}. {leak['type'].upper()}: {leak['module']}")
            print(f"     Reason: {leak['reason']}")
            if "files" in leak:
                print(f"     Files: {', '.join(leak['files'])}")
            if "exports" in leak and len(leak["exports"]) <= 10:
                print(f"     Exports: {', '.join(leak['exports'])}")
    else:
        print("\nPOTENTIAL INTERFACE LEAKS: None detected ✓")

    # Recommendations
    print("\nRECOMMENDATIONS:")

    missing_all_count = (
        summary["total_submodules"] - summary["submodules_with_explicit_all"]
    )
    if missing_all_count > 0:
        print(
            f"  1. Add __all__ declarations to {missing_all_count} submodules without them"
        )

    if summary["top_level_exports"] > 15:
        print(
            f"  2. Consider reducing top-level exports ({summary['top_level_exports']} current)"
        )

    if analysis["potential_leaks"]:
        print(
            f"  3. Review and fix {len(analysis['potential_leaks'])} potential interface leaks"
        )

    # Minimal API suggestion
    core_api = [
        export
        for export in analysis["top_level"]["explicit_exports"]
        if export not in ["clustering", "config", "optimization", "post_optimization", "utils"]
    ]

    print("\nMINIMAL PUBLIC API SUGGESTION:")
    print("  Consider exposing only these core functions at top level:")
    for api in sorted(core_api):
        print(f"    • {api}")
    print("  Keep internal modules private or require explicit imports")


def main():
    """Main entry point."""
    try:
        project_root = find_project_root()
        print(f"Analyzing FleetMix package at: {project_root}")

        analysis = analyze_package_exports(project_root)
        print_analysis(analysis)

        # Optionally write detailed results to file
        import json

        output_file = project_root / "interface_analysis.json"
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis written to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
