import os
from pathlib import Path

_project_root_cache: Path | None = None


def get_project_root() -> Path:
    global _project_root_cache
    if _project_root_cache is not None:
        return _project_root_cache

    # 1. Check environment variable (allows override)
    env_root_str = os.environ.get("FLEETMIX_PROJECT_ROOT")
    if env_root_str:
        env_root = Path(env_root_str).resolve()
        # Basic check for a valid project root marker
        if (
            (env_root / "pyproject.toml").exists()
            or (env_root / ".git").is_dir()
            or (env_root / "src" / "fleetmix").is_dir()
        ):
            _project_root_cache = env_root
            return env_root

    # 2. Auto-detection using a marker file (pyproject.toml is good)
    # Start from a known point within the src/fleetmix structure
    current_path = Path(__file__).resolve().parent  # Starts in src/fleetmix/utils

    for _ in range(
        5
    ):  # Limit search depth (src/fleetmix/utils -> src/fleetmix -> src -> project_root)
        if (current_path / "pyproject.toml").exists():
            _project_root_cache = current_path
            return current_path
        if current_path.parent == current_path:  # Reached filesystem root
            break
        current_path = current_path.parent

    raise FileNotFoundError(
        "Project root with 'pyproject.toml' could not be determined from utils. "
        "Searched for 'pyproject.toml', '.git', or 'src/fleetmix'. "
        "Consider setting the FLEETMIX_PROJECT_ROOT environment variable."
    )


PROJECT_ROOT = get_project_root()
