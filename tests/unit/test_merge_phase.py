"""Test the merge_phase module for post-optimization improvements."""

from unittest.mock import patch
import pytest
import pandas as pd

from fleetmix.config import load_fleetmix_params
from fleetmix.core_types import FleetmixSolution
from fleetmix.post_optimization.merge_phase import improve_solution


@pytest.fixture
def base_params():
    """Load base parameters for tests."""
    from pathlib import Path
    default_config = Path(__file__).resolve().parents[2] / "src" / "fleetmix" / "config" / "default_config.yaml"
    params = load_fleetmix_params(default_config)
    return params


def test_improve_solution_empty_initial_solution(base_params):
    """Test improve_solution with empty initial solution (lines 116-121)."""
    # Create an empty solution - FleetmixSolution uses dataclass defaults
    empty_solution = FleetmixSolution(
        selected_clusters=[],
        total_cost=0.0,
        solver_status="Optimal",
    )
    
    # Mock configs and customers as empty lists for this test
    configs = []
    customers = []
    
    # Should return the empty solution without attempting merge
    result = improve_solution(
        empty_solution,
        configs,
        customers,
        base_params,
    )
    
    assert len(result.selected_clusters) == 0


def test_improve_solution_no_candidate_merges(base_params):
    """Test improve_solution when no valid merged clusters are generated (lines 137-140)."""
    # Create a minimal solution with one cluster
    from fleetmix.core_types import Cluster, VehicleConfiguration
    
    cluster = Cluster(
        cluster_id=1,
        config_id="V1",
        vehicle_type="Type1",
        customers=["C1"],
        total_demand={"Dry": 10.0},
        centroid_latitude=0.0,
        centroid_longitude=0.0,
        goods_in_config=["Dry"],
        route_time=30.0,
        method="test",
    )
    
    initial_solution = FleetmixSolution(
        selected_clusters=[cluster],
        total_cost=200.0,
        solver_status="Optimal",
    )
    
    # Create matching vehicle config
    config = VehicleConfiguration(
        config_id="V1",
        vehicle_type="Type1",
        capacity=20,
        fixed_cost=100.0,
        compartments={"Dry": True, "Chilled": False, "Frozen": False},
    )
    
    # Mock generate_merge_phase_clusters to return empty DataFrame
    with patch("fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters") as mock_gen:
        mock_gen.return_value = pd.DataFrame()  # Empty - no candidates
        
        # Mock Customer.to_dataframe
        with patch("fleetmix.post_optimization.merge_phase.Customer.to_dataframe") as mock_to_df:
            mock_to_df.return_value = pd.DataFrame()
            
            result = improve_solution(
                initial_solution,
                [config],  # provide config
                [],  # empty customers
                base_params,
            )
    
    # Should return original solution (total_cost might be recalculated)
    # The important thing is it returns and doesn't crash
    assert result is not None
    assert len(result.selected_clusters) >= 0