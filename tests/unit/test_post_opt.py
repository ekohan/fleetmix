"""Tests for the post-optimization merge phase logic."""

from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from fleetmix.config import load_fleetmix_params
from fleetmix.core_types import Cluster, Customer, FleetmixSolution, VehicleConfiguration
from fleetmix.post_optimization.merge_phase import improve_solution


def _make_vehicle_config() -> list[VehicleConfiguration]:
    return [
        VehicleConfiguration(
            config_id=1,
            vehicle_type="TestVan",
            capacity=1000,
            fixed_cost=100,
            compartments={"Dry": True, "Chilled": True, "Frozen": True},
        )
    ]


def _make_customers() -> list[Customer]:
    data = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2"],
            "Latitude": [0.0, 1.0],
            "Longitude": [0.0, 1.0],
            "Dry_Demand": [10.0, 15.0],
            "Chilled_Demand": [0.0, 0.0],
            "Frozen_Demand": [0.0, 0.0],
        }
    )
    return Customer.from_dataframe(data)


def _make_cluster(cluster_id: str) -> Cluster:
    df = pd.DataFrame(
        [
            {
                "Cluster_ID": cluster_id,
                "Config_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 10.0},
                "Centroid_Latitude": 0.0,
                "Centroid_Longitude": 0.0,
                "Route_Time": 1.0,
                "Method": "test",
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ]
    )
    return Cluster.from_dataframe(df)[0]


def test_improve_solution_accepts_better_trial(tmp_path):
    """If optimise_fleet returns a cheaper selection, it becomes the new solution."""

    base_params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = replace(
        base_params,
        algorithm=replace(
            base_params.algorithm,
            max_improvement_iterations=2,
            small_cluster_size=5,
            nearest_merge_candidates=1,
        ),
    )

    initial_cluster = _make_cluster("orig")
    customers = _make_customers()
    configs = _make_vehicle_config()

    improved_cluster = replace(initial_cluster, cluster_id="merged")

    with patch(
        "fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters",
        return_value=pd.DataFrame(
            {
                "Cluster_ID": ["merged"],
                "Config_ID": [1],
                "Customers": [["C1", "C2"]],
                "Total_Demand": [{"Dry": 20.0}],
                "Route_Time": [2.0],
                "Centroid_Latitude": [0.5],
                "Centroid_Longitude": [0.5],
                "Method": ["merged"],
                "Dry": [1],
                "Chilled": [0],
                "Frozen": [0],
            }
        ),
    ) as mock_generate, patch(
        "fleetmix.post_optimization.merge_phase.optimize_fleet"
    ) as mock_optimize:
        mock_optimize.return_value = FleetmixSolution(
            selected_clusters=[improved_cluster],
            total_fixed_cost=50.0,
            total_variable_cost=0.0,
            total_penalties=0.0,
            solver_status="Optimal",
        )

        result = improve_solution(
            FleetmixSolution(
                selected_clusters=[initial_cluster],
                total_fixed_cost=100.0,
                total_variable_cost=0.0,
                total_penalties=0.0,
                solver_status="Optimal",
            ),
            configs,
            customers,
            params,
        )

    assert mock_generate.call_count >= 1
    assert mock_optimize.call_count >= 1
    assert result.total_cost == 50.0


def test_improve_solution_short_circuits_on_empty_selection():
    """When the solution has no selected clusters the merge loop should exit immediately."""

    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = replace(params, algorithm=replace(params.algorithm, max_improvement_iterations=2))

    with patch("fleetmix.post_optimization.merge_phase.logger") as mock_logger:
        outcome = improve_solution(
            FleetmixSolution(selected_clusters=[], total_cost=0.0),
            _make_vehicle_config(),
            _make_customers(),
            params,
        )

    assert outcome.selected_clusters == []
    mock_logger.info.assert_called_once()


def test_improve_solution_returns_original_when_no_candidates():
    """If no merge candidates are produced the original solution is returned."""

    params = load_fleetmix_params("src/fleetmix/config/default_config.yaml")
    params = replace(params, algorithm=replace(params.algorithm, max_improvement_iterations=1))
    original = FleetmixSolution(selected_clusters=[_make_cluster("orig")], total_cost=75.0)

    with patch(
        "fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters",
        return_value=pd.DataFrame(),
    ) as mock_generate:
        outcome = improve_solution(
            original,
            _make_vehicle_config(),
            _make_customers(),
            params,
        )

    mock_generate.assert_called_once()
    assert outcome.total_cost == original.total_cost
    assert outcome.selected_clusters == original.selected_clusters
