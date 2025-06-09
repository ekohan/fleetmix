from pathlib import Path

import pytest

# Define project root for path fixtures
repo_root = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def small_vrp_path():
    """Path to a small CVRP instance file for component tests"""
    return (
        repo_root
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "cvrp"
        / "X-n101-k25.vrp"
    )


@pytest.fixture(scope="session")
def small_sol_path():
    """Path to the solution file for the small CVRP instance"""
    return (
        repo_root
        / "src"
        / "fleetmix"
        / "benchmarking"
        / "datasets"
        / "cvrp"
        / "X-n101-k25.sol"
    )


@pytest.fixture(scope="session")
def mini_yaml():
    """Path to the minimal YAML config for integration tests"""
    return repo_root / "tests" / "_assets" / "smoke" / "mini.yaml"


@pytest.fixture(scope="session")
def mini_demand_csv():
    """Path to the minimal demand CSV for integration tests"""
    return repo_root / "tests" / "_assets" / "smoke" / "mini_demand.csv"


@pytest.fixture(autouse=True)
def tmp_results_dir(tmp_path, monkeypatch):
    """Redirect the project's results directory into a temp folder for integration tests"""
    # Create fake results dir
    fake_results = tmp_path / "results"
    fake_results.mkdir()
    # Run CLI commands from project root, but ensure results go to fake_results
    monkeypatch.setenv("PROJECT_RESULTS_DIR", str(fake_results))
    # Also redirect working directory to project root just in case
    monkeypatch.chdir(repo_root)
    return fake_results


# Toy data fixtures for FSM tests
@pytest.fixture
def toy_fsm_core_data():
    import pandas as pd

    from fleetmix.config.parameters import Parameters

    clusters_df = pd.DataFrame(
        {
            "Cluster_ID": [1],
            "Customers": [["C1"]],
            "Total_Demand": [{"Dry": 1, "Chilled": 0, "Frozen": 0}],
            "Config_ID": [1],
            "Centroid_Latitude": [0.0],
            "Centroid_Longitude": [0.0],
            "Route_Time": [1.0],
            "Method": ["test"],
        }
    )
    config_df = pd.DataFrame(
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "X",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ]
    )
    customers_df = pd.DataFrame(
        [
            {
                "Customer_ID": "C1",
                "Dry_Demand": 0,
                "Chilled_Demand": 0,
                "Frozen_Demand": 0,
            }
        ]
    )
    params = Parameters.from_yaml()
    return clusters_df, config_df, customers_df, params


@pytest.fixture
def toy_fsm_edge_data():
    import pandas as pd

    from fleetmix.config.parameters import Parameters

    clusters_df = pd.DataFrame(
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1", "C2"],
                "Total_Demand": {"Dry": 2, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ]
    )
    config_df = pd.DataFrame(
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 5,
                "Fixed_Cost": 10,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ]
    )
    params = Parameters.from_yaml()
    return clusters_df, config_df, params


# Toy data fixture for FSM model build tests
@pytest.fixture
def toy_fsm_model_build_data():
    import pandas as pd

    from fleetmix.config.parameters import Parameters

    clusters_df = pd.DataFrame(
        [
            {
                "Cluster_ID": "k1",
                "Customers": [1, 2],
                "Total_Demand": {"Dry": 5, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ]
    )
    config_df = pd.DataFrame(
        [
            {
                "Config_ID": "v1",
                "Capacity": 10,
                "Fixed_Cost": 100,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ]
    )
    params = Parameters.from_yaml("src/fleetmix/config/default_config.yaml")
    return clusters_df, config_df, params


# Smoke dataset fixtures
@pytest.fixture(scope="session")
def smoke_cluster_data():
    """Load the canonical smoke cluster configuration from YAML."""
    from pathlib import Path

    import yaml

    path = Path(__file__).parent / "_assets" / "smoke" / "mini.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def smoke_demand_data():
    """Load the canonical smoke demand data from CSV."""
    import csv
    from pathlib import Path

    path = Path(__file__).parent / "_assets" / "smoke" / "mini_demand.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)
