"""Test project_root utility functions"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import fleetmix.utils.project_root as project_root_module
from fleetmix.utils.project_root import get_project_root


class TestProjectRoot:
    """Test project root detection functionality"""
    
    def setup_method(self):
        """Reset the cache before each test"""
        project_root_module._project_root_cache = None
    
    def test_get_project_root_from_cache(self):
        """Test that cached value is returned if available"""
        fake_path = Path("/fake/cached/path")
        project_root_module._project_root_cache = fake_path
        
        result = get_project_root()
        assert result == fake_path
    
    def test_get_project_root_from_env_var(self, tmp_path):
        """Test project root detection from FLEETMIX_PROJECT_ROOT env var"""
        # Create a fake project structure
        (tmp_path / "pyproject.toml").touch()
        
        with patch.dict(os.environ, {'FLEETMIX_PROJECT_ROOT': str(tmp_path)}):
            result = get_project_root()
            assert result == tmp_path.resolve()
    
    def test_get_project_root_from_env_var_with_git(self, tmp_path):
        """Test env var validation with .git directory"""
        # Create .git directory instead of pyproject.toml
        (tmp_path / ".git").mkdir()
        
        with patch.dict(os.environ, {'FLEETMIX_PROJECT_ROOT': str(tmp_path)}):
            result = get_project_root()
            assert result == tmp_path.resolve()
    
    def test_get_project_root_from_env_var_with_src_fleetmix(self, tmp_path):
        """Test env var validation with src/fleetmix directory"""
        # Create src/fleetmix directory structure
        (tmp_path / "src" / "fleetmix").mkdir(parents=True)
        
        with patch.dict(os.environ, {'FLEETMIX_PROJECT_ROOT': str(tmp_path)}):
            result = get_project_root()
            assert result == tmp_path.resolve()
    
    def test_get_project_root_invalid_env_var(self, tmp_path):
        """Test that invalid env var is ignored and falls back to auto-detection"""
        # Don't create any marker files
        with patch.dict(os.environ, {'FLEETMIX_PROJECT_ROOT': str(tmp_path)}):
            # Mock the auto-detection path
            with patch('pathlib.Path.exists') as mock_exists:
                with patch('pathlib.Path.is_dir') as mock_is_dir:
                    mock_exists.return_value = False
                    mock_is_dir.return_value = False
                    
                    # Should raise because env var is invalid and auto-detection fails
                    with pytest.raises(FileNotFoundError, match="Project root"):
                        get_project_root()
    
    def test_get_project_root_reaches_filesystem_root(self):
        """Test error when reaching filesystem root without finding marker"""
        # Clear any environment variable
        with patch.dict(os.environ, {'FLEETMIX_PROJECT_ROOT': ''}):
            # Mock Path to simulate the filesystem root condition
            with patch('fleetmix.utils.project_root.Path') as MockPath:
                # Create a mock path that behaves like we're at the filesystem root
                mock_file_path = MagicMock()
                mock_current = MagicMock()
                
                # Set up the chain: __file__ -> resolve() -> parent (multiple times)
                MockPath.return_value = mock_file_path
                mock_file_path.resolve.return_value.parent = mock_current
                
                # No pyproject.toml exists anywhere
                mock_current.__truediv__.return_value.exists.return_value = False
                
                # Simulate reaching filesystem root (parent == self)
                mock_current.parent = mock_current
                
                with pytest.raises(FileNotFoundError, match="Project root"):
                    get_project_root() 