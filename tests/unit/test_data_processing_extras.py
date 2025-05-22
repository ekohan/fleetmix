import pandas as pd
import warnings
import pytest
from pathlib import Path

from fleetmix.utils.data_processing import load_customer_demand


def test_happy_path_and_zero_to_one(tmp_path, monkeypatch):
    # Prepare a CSV with mixed and zero demands
    csv = tmp_path / "test.csv"
    df = pd.DataFrame([
        {'ClientID':'C1','Lat':1.0,'Lon':2.0,'Kg':5,'ProductType':'Dry'},
        {'ClientID':'C1','Lat':1.0,'Lon':2.0,'Kg':0,'ProductType':'Chilled'},
        {'ClientID':'C1','Lat':1.0,'Lon':2.0,'Kg':0,'ProductType':'Frozen'},
        {'ClientID':'C2','Lat':3.0,'Lon':4.0,'Kg':0,'ProductType':'Dry'},
        {'ClientID':'C2','Lat':3.0,'Lon':4.0,'Kg':0,'ProductType':'Chilled'},
        {'ClientID':'C2','Lat':3.0,'Lon':4.0,'Kg':0,'ProductType':'Frozen'},
    ])
    df.to_csv(csv, index=False)
    # Monkeypatch demand profiles directory
    monkeypatch.setattr('fleetmix.utils.data_processing.get_demand_profiles_dir', lambda: tmp_path)

    # Capture pandas SettingWithCopyWarning if any
    with warnings.catch_warnings(record=True) as recs:
        warnings.simplefilter("always")
        out = load_customer_demand('test.csv')

    # No SettingWithCopyWarning should be emitted
    assert not any(w.category is pd.errors.SettingWithCopyWarning for w in recs)

    # Verify output columns
    expected_cols = {'Customer_ID','Latitude','Longitude','Dry_Demand','Chilled_Demand','Frozen_Demand'}
    assert expected_cols.issubset(set(out.columns))

    # Customer C1: Dry=5, Chilled=0, Frozen=0
    r1 = out[out['Customer_ID']=='C1'].iloc[0]
    assert r1['Dry_Demand'] == 5
    assert r1['Chilled_Demand'] == 0
    assert r1['Frozen_Demand'] == 0

    # Customer C2: all zero demands, Dry should reset to 1
    r2 = out[out['Customer_ID']=='C2'].iloc[0]
    assert r2['Dry_Demand'] == 1
    assert r2['Chilled_Demand'] == 0
    assert r2['Frozen_Demand'] == 0


def test_malformed_rows(tmp_path, monkeypatch):
    # Prepare a malformed CSV (wrong headers)
    csv = tmp_path / "bad.csv"
    csv.write_text("wrong,columns\n1,2\n")
    monkeypatch.setattr('fleetmix.utils.data_processing.get_demand_profiles_dir', lambda: tmp_path)

    # Expect an error when loading malformed data
    with pytest.raises(Exception):
        load_customer_demand('bad.csv') 