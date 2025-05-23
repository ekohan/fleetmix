#!/bin/bash

set -e  # Exit on error

# Check if the virtual environment already exists
if [ ! -d "fleetmix-env" ]; then
  echo "Creating Python virtual environment..."
  python -m venv fleetmix-env
else
  echo "Virtual environment 'fleetmix-env' already exists. Skipping creation."
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."

# Check the OS and set the activation command accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # For Windows
  source fleetmix-env/Scripts/activate
else
  # For Unix-based systems (Linux, macOS)
  source fleetmix-env/bin/activate
fi

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo "Virtual environment activated."
else
  echo "Failed to activate virtual environment. Exiting."
  exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install the fleetmix package in editable mode
echo "Installing fleetmix package in editable mode..."
pip install -e .

# Create database and import sales data
echo "Creating database and importing sales data..."
python data/import_data.py

# All done!
echo "Project setup complete!"