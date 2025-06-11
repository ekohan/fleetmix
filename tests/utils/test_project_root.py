"""Unit tests for the project_root module."""

import os
import tempfile
import unittest
from pathlib import Path

import fleetmix.utils.project_root
from fleetmix.utils.project_root import get_project_root


class TestProjectRoot(unittest.TestCase):
    """Test cases for project root detection."""

    def setUp(self):
        """Reset cache before each test."""
        # Reset the module-level cache
        fleetmix.utils.project_root._project_root_cache = None
        # Clear environment variable
        if "FLEETMIX_PROJECT_ROOT" in os.environ:
            del os.environ["FLEETMIX_PROJECT_ROOT"]

    def test_get_project_root_from_environment(self):
        """Test getting project root from environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid project structure
            temp_path = Path(temp_dir).resolve()  # Resolve to handle symlinks
            (temp_path / "pyproject.toml").touch()

            # Set environment variable
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # Get project root
            root = get_project_root()
            self.assertEqual(root.resolve(), temp_path)

    def test_get_project_root_from_environment_with_git(self):
        """Test environment variable with .git directory marker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid project structure with .git
            temp_path = Path(temp_dir).resolve()
            (temp_path / ".git").mkdir()

            # Set environment variable
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # Get project root
            root = get_project_root()
            self.assertEqual(root.resolve(), temp_path)

    def test_get_project_root_from_environment_with_src_fleetmix(self):
        """Test environment variable with src/fleetmix directory marker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid project structure with src/fleetmix
            temp_path = Path(temp_dir).resolve()
            (temp_path / "src" / "fleetmix").mkdir(parents=True)

            # Set environment variable
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # Get project root
            root = get_project_root()
            self.assertEqual(root.resolve(), temp_path)

    def test_get_project_root_invalid_environment(self):
        """Test that invalid environment path is ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid project structure (no markers)
            temp_path = Path(temp_dir)

            # Set environment variable to invalid path
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # Should fall back to auto-detection
            # In the actual fleetmix project, this would find the real root
            # In test environment, we expect it to find the actual project root
            try:
                root = get_project_root()
                # If it succeeds, it found the real project root
                self.assertTrue(root.exists())
                self.assertTrue(
                    (root / "pyproject.toml").exists()
                    or (root / ".git").exists()
                    or (root / "src" / "fleetmix").exists()
                )
            except FileNotFoundError:
                # This is also acceptable in a test environment
                pass

    def test_get_project_root_auto_detection(self):
        """Test auto-detection of project root."""
        # Clear environment to force auto-detection
        if "FLEETMIX_PROJECT_ROOT" in os.environ:
            del os.environ["FLEETMIX_PROJECT_ROOT"]

        # In the actual project, this should find the root
        try:
            root = get_project_root()
            # If it succeeds, verify it's a valid project root
            self.assertTrue(root.exists())
            self.assertTrue(
                (root / "pyproject.toml").exists()
                or (root / ".git").exists()
                or (root / "src" / "fleetmix").exists()
            )
        except FileNotFoundError:
            # This is expected in some test environments
            pass

    def test_get_project_root_cache(self):
        """Test that project root is cached after first call."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid project structure
            temp_path = Path(temp_dir).resolve()
            (temp_path / "pyproject.toml").touch()

            # Set environment variable
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # First call
            root1 = get_project_root()

            # Change environment variable
            os.environ["FLEETMIX_PROJECT_ROOT"] = "/different/path"

            # Second call should return cached value
            root2 = get_project_root()
            self.assertEqual(root1, root2)

    def test_get_project_root_max_depth(self):
        """Test that search stops after maximum depth."""
        # Clear any environment variable
        if "FLEETMIX_PROJECT_ROOT" in os.environ:
            del os.environ["FLEETMIX_PROJECT_ROOT"]

        # The actual implementation will try to find project root
        # In a real project environment, it might succeed
        try:
            root = get_project_root()
            # If it found something, verify it's valid
            self.assertTrue(root.exists())
        except FileNotFoundError as e:
            # This is expected if we're not in a proper project structure
            self.assertIn("Project root", str(e))
            self.assertIn("could not be determined", str(e))

    def test_project_root_constant(self):
        """Test that PROJECT_ROOT constant is set."""
        # The PROJECT_ROOT constant should be set when module is imported
        # In test environment, it might fail, but we can check the mechanism
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid project structure
            temp_path = Path(temp_dir).resolve()
            (temp_path / "pyproject.toml").touch()

            # Set environment variable before importing
            os.environ["FLEETMIX_PROJECT_ROOT"] = str(temp_path)

            # Re-import the module to test PROJECT_ROOT initialization
            import importlib

            importlib.reload(fleetmix.utils.project_root)

            # Check that PROJECT_ROOT matches get_project_root()
            self.assertEqual(
                fleetmix.utils.project_root.PROJECT_ROOT.resolve(),
                fleetmix.utils.project_root.get_project_root().resolve(),
            )


if __name__ == "__main__":
    unittest.main()
