#!/usr/bin/env python3
"""
FleetMix Release Automation
Handles versioning, changelog generation, and release process.
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import toml  # type: ignore


def get_current_version() -> str:
    """Extract current version from pyproject.toml."""
    with open("pyproject.toml") as f:
        data = toml.load(f)
        return str(data["project"]["version"])


def analyze_commits_since_tag(tag: Optional[str] = None) -> dict:
    """Analyze commits to determine version bump and changelog."""
    if tag:
        cmd = ["git", "log", f"{tag}..HEAD", "--oneline"]
    else:
        cmd = ["git", "log", "--oneline", "-20"]  # Last 20 if no tag

    result = subprocess.run(cmd, capture_output=True, text=True)

    commits: dict[str, list[tuple[str, str]]] = {
        "breaking": [],
        "features": [],
        "fixes": [],
        "perf": [],
        "docs": [],
        "other": [],
    }

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        commit_hash = line.split()[0]
        message = " ".join(line.split()[1:])

        # Conventional commit parsing
        if message.startswith("BREAKING:") or "BREAKING CHANGE:" in message:
            commits["breaking"].append((commit_hash, message))
        elif message.startswith("feat:") or message.startswith("feature:"):
            commits["features"].append((commit_hash, message))
        elif message.startswith("fix:") or message.startswith("bugfix:"):
            commits["fixes"].append((commit_hash, message))
        elif message.startswith("perf:") or message.startswith("performance:"):
            commits["perf"].append((commit_hash, message))
        elif message.startswith("docs:"):
            commits["docs"].append((commit_hash, message))
        else:
            commits["other"].append((commit_hash, message))

    return commits


def determine_version_bump(commits: dict, current: str) -> Tuple[str, str]:
    """Determine new version based on conventional commits."""
    major, minor, patch = map(int, current.split("."))

    if commits["breaking"]:
        return f"{major + 1}.0.0", "major"
    elif commits["features"]:
        return f"{major}.{minor + 1}.0", "minor"
    elif commits["fixes"] or commits["perf"]:
        return f"{major}.{minor}.{patch + 1}", "patch"
    else:
        return f"{major}.{minor}.{patch + 1}", "patch"


def generate_changelog(commits: dict, version: str) -> str:
    """Generate changelog in markdown format."""
    date = datetime.now().strftime("%Y-%m-%d")
    changelog = f"## [{version}] - {date}\n\n"

    if commits["breaking"]:
        changelog += "### ⚠️ BREAKING CHANGES\n"
        for hash, msg in commits["breaking"]:
            # Clean up commit message
            msg = msg.replace("BREAKING:", "").replace("BREAKING CHANGE:", "").strip()
            changelog += f"- {msg} ({hash[:7]})\n"
        changelog += "\n"

    if commits["features"]:
        changelog += "### ✨ Features\n"
        for hash, msg in commits["features"]:
            msg = msg.replace("feat:", "").replace("feature:", "").strip()
            changelog += f"- {msg} ({hash[:7]})\n"
        changelog += "\n"

    if commits["fixes"]:
        changelog += "### 🐛 Bug Fixes\n"
        for hash, msg in commits["fixes"]:
            msg = msg.replace("fix:", "").replace("bugfix:", "").strip()
            changelog += f"- {msg} ({hash[:7]})\n"
        changelog += "\n"

    if commits["perf"]:
        changelog += "### ⚡ Performance\n"
        for hash, msg in commits["perf"]:
            msg = msg.replace("perf:", "").replace("performance:", "").strip()
            changelog += f"- {msg} ({hash[:7]})\n"
        changelog += "\n"

    return changelog


def update_version_in_files(old_version: str, new_version: str) -> None:
    """Update version in all relevant files."""
    # Update pyproject.toml using TOML parsing
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        with open(pyproject_path) as f:
            data = toml.load(f)

        data["project"]["version"] = new_version

        with open(pyproject_path, "w") as f:
            toml.dump(data, f)
        print(f"  ✅ Updated {pyproject_path}")

    # Update __init__.py using string replacement (appropriate for Python files)
    init_path = Path("src/fleetmix/__init__.py")
    if init_path.exists():
        old_text = f'__version__ = "{old_version}"'
        new_text = f'__version__ = "{new_version}"'

        with open(init_path, "r") as f:
            content = f.read()

        if old_text in content:
            content = content.replace(old_text, new_text)
            with open(init_path, "w") as f:
                f.write(content)
            print(f"  ✅ Updated {init_path}")
        else:
            print(f"  ⚠️  Could not find version string in {init_path}")


def update_citation_file(version: str) -> None:
    """Update CITATION.cff with new version and date."""
    citation_file = Path("CITATION.cff")
    if not citation_file.exists():
        print("  ⚠️  No CITATION.cff found")
        return

    with open(citation_file) as f:
        content = f.read()

    # Update version
    content = re.sub(r"version: .*", f"version: {version}", content)

    # Update date
    date = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(r"date-released: .*", f"date-released: {date}", content)

    with open(citation_file, "w") as f:
        f.write(content)

    print("  ✅ Updated CITATION.cff")


def run_tests() -> bool:
    """Run full test suite."""
    print("\n🧪 Running full test suite...")
    result = subprocess.run(["uv", "run", "pytest", "--quiet"], capture_output=True)
    return result.returncode == 0


def create_git_tag(version: str, message: str) -> None:
    """Create annotated git tag."""
    subprocess.run(["git", "tag", "-a", f"v{version}", "-m", message], check=True)
    print(f"  ✅ Created tag v{version}")


def build_package() -> None:
    """Build Python package."""
    print("\n📦 Building package...")
    subprocess.run(["uv", "build"], check=True)
    print("  ✅ Package built")


def main(dry_run: bool = False) -> int:
    """Run the release process."""
    print("🚀 FleetMix Release Automation")
    print("=" * 40)

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Get last tag
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True
    )
    last_tag = result.stdout.strip() if result.returncode == 0 else None

    # Analyze commits
    print("\n📝 Analyzing commits...")
    commits = analyze_commits_since_tag(last_tag)

    # Determine new version
    new_version, bump_type = determine_version_bump(commits, current_version)
    print(f"Version bump: {current_version} → {new_version} ({bump_type})")

    # Generate changelog
    changelog = generate_changelog(commits, new_version)
    print("\n📋 Changelog:")
    print(changelog)

    if dry_run:
        print("\n🔍 DRY RUN - No changes made")
        return 0

    # Confirm with user
    response = input("\n⚠️  Proceed with release? [y/N]: ")
    if response.lower() != "y":
        print("❌ Release cancelled")
        return 1

    # Run tests
    if not run_tests():
        print("❌ Tests failed - aborting release")
        return 1
    print("  ✅ All tests passed")

    # Update version in files
    print("\n📝 Updating version in files...")
    update_version_in_files(current_version, new_version)
    update_citation_file(new_version)

    # Update CHANGELOG.md
    changelog_file = Path("CHANGELOG.md")
    if changelog_file.exists():
        with open(changelog_file) as f:
            existing = f.read()

        # Insert new changelog after header
        if "# Changelog" in existing:
            parts = existing.split("# Changelog", 1)
            new_content = f"# Changelog\n\n{changelog}{parts[1]}"
        else:
            new_content = f"# Changelog\n\n{changelog}\n{existing}"

        with open(changelog_file, "w") as f:
            f.write(new_content)
        print("  ✅ Updated CHANGELOG.md")

    # Commit changes
    print("\n📝 Committing changes...")
    subprocess.run(
        [
            "git",
            "add",
            "pyproject.toml",
            "CHANGELOG.md",
            "CITATION.cff",
            "src/fleetmix/__init__.py",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"Release v{new_version}\n\n{changelog}"], check=True
    )

    # Create tag
    create_git_tag(new_version, f"Release v{new_version}")

    # Build package
    build_package()

    print("\n✅ Release prepared successfully!")
    print("\nNext steps:")
    print("1. Push changes: git push origin main --tags")
    print("2. Upload to PyPI: uv publish")
    print("3. Create GitHub release from tag")

    return 0


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    sys.exit(main(dry_run))
