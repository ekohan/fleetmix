import pandas as pd

from fleetmix.utils.data_processing import load_customer_demand


def test_load_customer_demand(tmp_path, monkeypatch):
    # Create a mini CSV file
    data = [
        {"ClientID": "C1", "Lat": 1.0, "Lon": 2.0, "Kg": 5, "ProductType": "Dry"},
        {"ClientID": "C1", "Lat": 1.0, "Lon": 2.0, "Kg": 3, "ProductType": "Chilled"},
        {"ClientID": "C2", "Lat": 2.0, "Lon": 3.0, "Kg": 0, "ProductType": "Frozen"},
    ]
    df = pd.DataFrame(data)
    profiles_dir = tmp_path / "demand_profiles"
    profiles_dir.mkdir()
    csv_path = profiles_dir / "test.csv"
    df.to_csv(csv_path, index=False)

    # Monkeypatch demand directory and file name
    monkeypatch.setattr(
        "fleetmix.utils.data_processing.get_demand_profiles_dir", lambda: profiles_dir
    )
    result = load_customer_demand("test.csv")

    # Check columns and dtypes
    assert "Customer_ID" in result.columns
    assert "Latitude" in result.columns
    assert "Longitude" in result.columns
    assert "Dry_Demand" in result.columns
    assert "Chilled_Demand" in result.columns
    assert "Frozen_Demand" in result.columns

    # C1 should have Dry=5+3? Actually pivot sums by ProductType separately
    row1 = result[result["Customer_ID"] == "C1"].iloc[0]
    assert row1["Dry_Demand"] == 5
    assert row1["Chilled_Demand"] == 3
    assert row1["Frozen_Demand"] == 0

    # C2: all zero demand => Dry_Demand reset to 1
    row2 = result[result["Customer_ID"] == "C2"].iloc[0]
    assert row2["Dry_Demand"] == 1
    assert row2["Chilled_Demand"] == 0
    assert row2["Frozen_Demand"] == 0
