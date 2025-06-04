import pandas as pd
import pytest
import sys

import fleetmix.post_optimization.merge_phase as merge_phase
import fleetmix.optimization
from fleetmix.config.parameters import Parameters
from fleetmix.internal_types import FleetmixSolution

# Helper to create a minimal clusters DataFrame with goods columns
def make_cluster_df(cluster_id):
    # Get the goods from parameters
    goods = Parameters.from_yaml().goods
    
    # Create demand dict for Total_Demand column
    demand_dict = {g: 1 for g in goods}
    
    return pd.DataFrame([{  
        'Cluster_ID': cluster_id,
        'Config_ID': 1,
        'Customers': [f'C{cluster_id}'],  # Add a Customers column
        'Total_Demand': demand_dict,      # Add Total_Demand with dict of goods
        'Route_Time': 10,                 # Add Route_Time column
        'Centroid_Latitude': 42.0,        # Add lat/lon for centroid
        'Centroid_Longitude': -71.0,
        'Method': 'test',                 # Add method field
        # include all goods by default
        **{g: 1 for g in goods}
    }])

# Create a minimal configurations DataFrame
def make_config_df():
    return pd.DataFrame([{
        'Config_ID': 1,
        'Fixed_Cost': 100,
        'Capacity': 1000,
        'Dry': True,
        'Chilled': True,
        'Frozen': True
    }])


def test_no_merges(monkeypatch):
    """Scenario B: no feasible merges => exit immediately without solving"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        return pd.DataFrame()  # always empty

    def fake_solve(combined, configurations_df, customers_df, params, solver=None, verbose=False):
        calls['solve'] += 1
        return FleetmixSolution(selected_clusters=make_cluster_df('m'), total_cost=50)

    # Patch the imported function directly 
    import fleetmix.post_optimization.merge_phase
    original_gen = fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters
    monkeypatch.setattr(fleetmix.post_optimization.merge_phase, "generate_merge_phase_clusters", fake_gen)
    
    # Patch the imported optimize_fleet_selection
    original_solve = fleetmix.optimization.optimize_fleet_selection
    monkeypatch.setattr(fleetmix.optimization, "optimize_fleet_selection", fake_solve)
    
    try:
        initial_clusters = make_cluster_df('c')
        initial_solution = FleetmixSolution(selected_clusters=initial_clusters, total_cost=100)
        params = Parameters.from_yaml()

        result = merge_phase.improve_solution(initial_solution, make_config_df(), pd.DataFrame(), params)
        assert result.total_cost == initial_solution.total_cost
        assert result.selected_clusters.equals(initial_solution.selected_clusters)
        assert calls['gen'] == 1
        assert calls['solve'] == 0
    finally:
        # Restore the original functions to avoid affecting other tests
        fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters = original_gen
        fleetmix.optimization.optimize_fleet_selection = original_solve


def test_single_merge_then_no_more(monkeypatch):
    """Scenario A: one merge lowers cost, next yields none"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        if calls['gen'] == 1:
            return make_cluster_df('m1')
        return pd.DataFrame()

    def fake_solve(combined, configurations_df, customers_df, params, solver=None, verbose=False):
        calls['solve'] += 1
        return FleetmixSolution(selected_clusters=make_cluster_df('m1'), total_cost=90)

    # Patch the imported function directly 
    import fleetmix.post_optimization.merge_phase
    original_gen = fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters
    monkeypatch.setattr(fleetmix.post_optimization.merge_phase, "generate_merge_phase_clusters", fake_gen)
    
    # Patch the imported optimize_fleet_selection
    original_solve = fleetmix.optimization.optimize_fleet_selection
    monkeypatch.setattr(fleetmix.optimization, "optimize_fleet_selection", fake_solve)
    
    try:
        initial_clusters = make_cluster_df('c1')
        initial_solution = FleetmixSolution(selected_clusters=initial_clusters, total_cost=100)
        params = Parameters.from_yaml()

        result = merge_phase.improve_solution(initial_solution, make_config_df(), pd.DataFrame(), params)
        assert result.total_cost == 90
        assert calls['gen'] == 2
        assert calls['solve'] == 1
    finally:
        # Restore the original functions to avoid affecting other tests
        fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters = original_gen
        fleetmix.optimization.optimize_fleet_selection = original_solve


def test_iteration_cap(monkeypatch):
    """Scenario C: always merge & improve => stops at iteration cap"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        # always return a dummy merge
        return make_cluster_df(f'g{calls["gen"]}')

    def fake_solve(combined, configurations_df, customers_df, params, solver=None, verbose=False):
        calls['solve'] += 1
        # decreasing cost each call
        cost = 100 - calls['solve']
        # ensure selected_clusters changes
        return FleetmixSolution(selected_clusters=make_cluster_df(f'g{calls["solve"]}'), total_cost=cost)

    # Patch the imported function directly 
    import fleetmix.post_optimization.merge_phase
    original_gen = fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters
    monkeypatch.setattr(fleetmix.post_optimization.merge_phase, "generate_merge_phase_clusters", fake_gen)
    
    # Patch the imported optimize_fleet_selection
    original_solve = fleetmix.optimization.optimize_fleet_selection
    monkeypatch.setattr(fleetmix.optimization, "optimize_fleet_selection", fake_solve)
    
    try:
        initial_clusters = make_cluster_df('c0')
        initial_solution = FleetmixSolution(selected_clusters=initial_clusters, total_cost=100)
        params = Parameters.from_yaml()
        params.max_improvement_iterations = 3

        result = merge_phase.improve_solution(initial_solution, make_config_df(), pd.DataFrame(), params)
        assert calls['solve'] == 3
        assert result.total_cost == 100 - 3
    finally:
        # Restore the original functions to avoid affecting other tests
        fleetmix.post_optimization.merge_phase.generate_merge_phase_clusters = original_gen
        fleetmix.optimization.optimize_fleet_selection = original_solve 