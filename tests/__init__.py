"""Test package initialization for coverage.

Import the CLI module eagerly so module-level registries are exercised when
collecting coverage data."""

import importlib

importlib.import_module("fleetmix.app")
