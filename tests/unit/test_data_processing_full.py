"""Test the data_processing module comprehensively."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from fleetmix.utils.data_processing import (
    data_dir,
    get_demand_profiles_dir,
    load_customer_demand
)


def test_data_dir():
    """Test the data_dir function returns a valid path."""
    path = data_dir()
    assert isinstance(path, Path)
    # Should end with 'data'
    assert path.name == 'data'


def test_get_demand_profiles_dir():
    """Test getting demand profiles directory."""
    path = get_demand_profiles_dir()
    assert isinstance(path, Path)
    # Should be inside data directory
    assert path.parent.name == 'data'
    assert path.name == 'demand_profiles'


def test_load_customer_demand_absolute_path(tmp_path):
    """Test loading customer demand with absolute path."""
    # Create test CSV file
    csv_content = """ClientID,Lat,Lon,Kg,ProductType
C001,10.5,20.5,100,Dry
C001,10.5,20.5,50,Chilled
C002,11.0,21.0,200,Frozen
C003,12.0,22.0,150,Dry
"""
    csv_file = tmp_path / "test_demand.csv"
    csv_file.write_text(csv_content)
    
    # Load using absolute path
    df = load_customer_demand(str(csv_file))
    
    # Verify structure
    assert len(df) == 3  # 3 unique customers
    assert set(df.columns) == {
        'Customer_ID', 'Latitude', 'Longitude',
        'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand'
    }
    
    # Verify data
    c001 = df[df['Customer_ID'] == 'C001'].iloc[0]
    assert c001['Dry_Demand'] == 100
    assert c001['Chilled_Demand'] == 50
    assert c001['Frozen_Demand'] == 0
    
    c002 = df[df['Customer_ID'] == 'C002'].iloc[0]
    assert c002['Dry_Demand'] == 0
    assert c002['Chilled_Demand'] == 0
    assert c002['Frozen_Demand'] == 200


def test_load_customer_demand_relative_path(tmp_path, monkeypatch):
    """Test loading customer demand with relative path."""
    # Create test CSV in current directory
    csv_content = """ClientID,Lat,Lon,Kg,ProductType
C001,10.5,20.5,100,Dry
C001,10.5,20.5,0,Chilled
C001,10.5,20.5,0,Frozen
"""
    csv_file = tmp_path / "relative_demand.csv"
    csv_file.write_text(csv_content)
    
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Load using relative path
    df = load_customer_demand("relative_demand.csv")
    
    assert len(df) == 1
    assert df.iloc[0]['Customer_ID'] == 'C001'
    assert df.iloc[0]['Dry_Demand'] == 100


def test_load_customer_demand_aggregation():
    """Test that multiple entries for same customer are aggregated."""
    test_file = Path(__file__).parent.parent / "_assets" / "test_data_aggregation.csv"
    df = load_customer_demand(str(test_file))
    
    # Should have only one row for C001
    assert len(df) == 1
    assert df.iloc[0]['Customer_ID'] == 'C001'
    assert df.iloc[0]['Dry_Demand'] == 150  # 100 + 50
    assert df.iloc[0]['Chilled_Demand'] == 75
    assert df.iloc[0]['Frozen_Demand'] == 0


def test_load_customer_demand_zero_demands(tmp_path):
    """Test handling of customers with zero demands."""
    # Create CSV with customer having zero demand
    csv_content = """ClientID,Lat,Lon,Kg,ProductType
C001,10.5,20.5,0,Dry
C001,10.5,20.5,0,Chilled
C001,10.5,20.5,0,Frozen
"""
    csv_file = tmp_path / "zero_demand.csv"
    csv_file.write_text(csv_content)
    
    df = load_customer_demand(str(csv_file))
    
    # Zero demands should be converted to 1 Dry_Demand
    assert len(df) == 1
    assert df.iloc[0]['Dry_Demand'] == 1
    assert df.iloc[0]['Chilled_Demand'] == 0
    assert df.iloc[0]['Frozen_Demand'] == 0


def test_load_customer_demand_latin1_encoding():
    """Test loading CSV with latin-1 encoded characters."""
    test_file = Path(__file__).parent.parent / "_assets" / "test_data_latin1.csv"
    df = load_customer_demand(str(test_file))
    assert len(df) == 1
    assert df.iloc[0]['Customer_ID'] == 'Coffee001'
    assert df.iloc[0]['Dry_Demand'] == 100
    assert df.iloc[0]['Chilled_Demand'] == 0
    assert df.iloc[0]['Frozen_Demand'] == 0


def test_load_customer_demand_missing_columns(tmp_path):
    """Test error handling for CSV with missing columns."""
    # Create CSV missing required columns
    csv_content = """ClientID,Lat,Kg
C001,10.5,100
"""
    csv_file = tmp_path / "missing_cols.csv"
    csv_file.write_text(csv_content)
    
    # Should raise an error
    with pytest.raises(KeyError):
        load_customer_demand(str(csv_file))


def test_load_customer_demand_partial_zero_demands(tmp_path):
    """Test mixed zero and non-zero demands."""
    csv_content = """ClientID,Lat,Lon,Kg,ProductType
C001,10.5,20.5,100,Dry
C002,11.0,21.0,0,Dry
C002,11.0,21.0,0,Chilled
C002,11.0,21.0,0,Frozen
C003,12.0,22.0,50,Frozen
"""
    csv_file = tmp_path / "mixed_demand.csv"
    csv_file.write_text(csv_content)
    
    df = load_customer_demand(str(csv_file))
    
    # C002 should have Dry_Demand set to 1
    c002 = df[df['Customer_ID'] == 'C002'].iloc[0]
    assert c002['Dry_Demand'] == 1
    assert c002['Chilled_Demand'] == 0
    assert c002['Frozen_Demand'] == 0
    
    # C001 and C003 should keep their original demands
    c001 = df[df['Customer_ID'] == 'C001'].iloc[0]
    assert c001['Dry_Demand'] == 100
    
    c003 = df[df['Customer_ID'] == 'C003'].iloc[0]
    assert c003['Frozen_Demand'] == 50 