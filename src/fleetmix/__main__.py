"""
Entry point for running fleetmix as a module: python -m fleetmix
"""
import sys
from fleetmix.app import app

if __name__ == "__main__":
    sys.exit(app()) 