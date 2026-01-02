#!/usr/bin/env python3
"""
FleetMix Quality Gates - Pre-commit automation
Run before each commit to ensure code quality standards.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def get_modified_python_files() -> List[Path]:
    """Get list of modified Python files in current commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return []

    files = []
    for line in result.stdout.strip().split("\n"):
        if line.endswith(".py") and Path(line).exists():
            files.append(Path(line))
    return files


def check_type_hints(file_path: Path) -> List[str]:
    """Check if functions have type hints."""
    issues = []
    with open(file_path) as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return [f"Syntax error in {file_path}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions and test functions
            if node.name.startswith("_") or node.name.startswith("test_"):
                continue

            # Check return type hint
            if node.returns is None and node.name != "__init__":
                issues.append(
                    f"{file_path}:{node.lineno} - Function '{node.name}' missing return type hint"
                )

            # Check parameter type hints
            for arg in node.args.args:
                if arg.annotation is None and arg.arg != "self":
                    issues.append(
                        f"{file_path}:{node.lineno} - Parameter '{arg.arg}' in '{node.name}' missing type hint"
                    )

    return issues


def check_test_coverage(file_path: Path) -> List[str]:
    """Check if source file has corresponding test file."""
    issues = []

    # Skip test files themselves
    if "test" in file_path.parts or file_path.name.startswith("test_"):
        return []

    # Skip non-source files
    if "src" not in file_path.parts:
        return []

    # Construct expected test file path
    relative_path = file_path.relative_to(Path("src"))
    test_file = Path("tests/unit") / f"test_{relative_path.name}"

    if not test_file.exists():
        issues.append(f"No test file found for {file_path} (expected: {test_file})")

    return issues


def check_yaml_configs() -> List[str]:
    """Validate YAML configuration files."""
    issues = []
    config_files = [f for f in Path(".").rglob("*") if f.suffix in {".yaml", ".yml"}]

    for config_file in config_files:
        # Skip CI configs
        if ".github" in config_file.parts:
            continue

        try:
            with open(config_file) as f:
                data = yaml.safe_load(f)

            # Check for FleetMix-specific config structure
            if config_file.name in ["config.yaml", "fleetmix.yaml"]:
                required_keys = {"vehicles", "goods", "optimization"}
                missing = required_keys - set(data.keys())
                if missing:
                    issues.append(f"{config_file}: Missing required keys: {missing}")
        except yaml.YAMLError as e:
            issues.append(f"{config_file}: Invalid YAML - {e}")

    return issues


def check_docstrings(file_path: Path) -> List[str]:
    """Check if public functions have docstrings."""
    issues = []
    with open(file_path) as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # Skip private and test items
            if node.name.startswith("_") or node.name.startswith("Test"):
                continue

            # Check for docstring
            if not ast.get_docstring(node):
                item_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                issues.append(
                    f"{file_path}:{node.lineno} - {item_type} '{node.name}' missing docstring"
                )

    return issues


def run_incremental_tests(modified_files: List[Path]) -> Tuple[bool, str]:
    """Run only tests for modified files."""
    if not modified_files:
        return True, "No Python files modified"

    # Find test files to run
    test_files = []
    for file in modified_files:
        if "test" in str(file):
            test_files.append(str(file))
        else:
            # Try to find corresponding test
            test_name = f"test_{file.stem}.py"
            test_path = Path("tests/unit") / test_name
            if test_path.exists():
                test_files.append(str(test_path))

    if not test_files:
        return True, "No tests to run for modified files"

    # Run tests
    result = subprocess.run(
        ["uv", "run", "pytest"] + test_files + ["--quiet"],
        capture_output=True,
        text=True,
    )

    return result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    """Run all quality checks."""
    print("🔍 FleetMix Quality Gates")
    print("-" * 40)

    all_issues = []

    # Get modified files
    modified_files = get_modified_python_files()

    if not modified_files:
        print("✅ No Python files modified")
        return 0

    print(f"Checking {len(modified_files)} modified Python file(s)...")

    # Run checks
    for file in modified_files:
        print(f"\n📄 {file}")

        # Type hints
        type_issues = check_type_hints(file)
        if type_issues:
            print("  ❌ Type hint issues:")
            for issue in type_issues:
                print(f"    - {issue}")
            all_issues.extend(type_issues)
        else:
            print("  ✅ Type hints OK")

        # Docstrings
        doc_issues = check_docstrings(file)
        if doc_issues:
            print("  ⚠️  Docstring warnings:")
            for issue in doc_issues:
                print(f"    - {issue}")
            # Don't block on docstrings, just warn
        else:
            print("  ✅ Docstrings OK")

        # Test coverage
        test_issues = check_test_coverage(file)
        if test_issues:
            print("  ⚠️  Test coverage:")
            for issue in test_issues:
                print(f"    - {issue}")
            # Don't block on missing tests, just warn

    # Check YAML configs
    yaml_issues = check_yaml_configs()
    if yaml_issues:
        print("\n❌ YAML configuration issues:")
        for issue in yaml_issues:
            print(f"  - {issue}")
        all_issues.extend(yaml_issues)

    # Run incremental tests
    print("\n🧪 Running tests for modified files...")
    tests_passed, test_output = run_incremental_tests(modified_files)
    if not tests_passed:
        print("❌ Tests failed:")
        print(test_output)
        all_issues.append("Tests failed")
    else:
        print("✅ Tests passed")

    # Final verdict
    print("\n" + "=" * 40)
    if all_issues:
        print(f"❌ Quality gates FAILED - {len(all_issues)} issue(s) found")
        print("\nTo commit anyway, use: git commit --no-verify")
        return 1
    else:
        print("✅ All quality gates PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
