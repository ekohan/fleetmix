"""
Simple end-to-end tests for benchmarking MCVRP and CVRP instances.
Tests focus on parsing, conversion, and basic solving with minimal fixtures.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

from fleetmix.benchmarking import (
    parse_mcvrp, CVRPParser,
    convert_mcvrp_to_fsm, convert_cvrp_to_fsm,
    CVRPBenchmarkType
)
from fleetmix.pipeline import VRPType, convert_to_fsm, run_optimization


class TestSimpleBenchmarking:
    """Simple end-to-end tests for VRP benchmarking."""
    
    def test_cvrp_simple_instance(self, tmp_path):
        """Test CVRP with a minimal 3-customer instance."""
        # Create a simple CVRP instance file
        cvrp_content = """NAME: test-3
COMMENT: Minimal test instance
TYPE: CVRP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
CAPACITY: 100
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
4 -1 0
DEMAND_SECTION
1 0
2 30
3 30
4 30
DEPOT_SECTION
1
-1
EOF"""
        
        # Write to temp file
        cvrp_file = tmp_path / "test-3.vrp"
        cvrp_file.write_text(cvrp_content)
        
        # Parse instance
        parser = CVRPParser(str(cvrp_file))
        instance = parser.parse()
        
        # Verify parsing
        assert instance.name == "test-3"
        assert instance.dimension == 4
        assert instance.capacity == 100
        
        # Convert to FSM format
        customers_df, params = convert_cvrp_to_fsm(
            instance_names=["test-3"],
            benchmark_type=CVRPBenchmarkType.NORMAL,
            num_goods=1,
            custom_instance_paths={"test-3": cvrp_file}
        )
        
        # Verify we have 3 customers
        assert len(customers_df) == 3
        
        # Override parameters for quick test
        params.time_limit_minutes = 0.1  # 6 seconds
        params.results_dir = tmp_path
        
        # Run optimization
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )
        
        # Verify we got a solution
        assert solution is not None
        assert solution.solver_status is not None
        assert solution.vehicles_used is not None
        
        # Check basic solution properties
        assert solution.solver_status in ['Optimal', 'Feasible', 'TimeLimit']
        
        # Verify cost components exist
        assert solution.total_fixed_cost is not None
        assert solution.total_variable_cost is not None
        
        # For simple instances, should use at least one vehicle
        if isinstance(solution.vehicles_used, dict):
            total_vehicles = sum(solution.vehicles_used.values())
            assert total_vehicles > 0

    def test_mcvrp_simple_instance(self, tmp_path):
        """Test MCVRP with a minimal 3-customer, 3-product instance."""
        # Create a simple MCVRP instance file
        mcvrp_content = """NAME: test-mcvrp-3
COMMENT: Minimal MCVRP test
TYPE: MCVRP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
CAPACITY: 100
VEHICLES: 2
PRODUCT TYPES: 3
COMPARTMENTS: 3
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
4 -1 0
DEMAND_SECTION
1 0 0 0
2 30 0 0
3 0 30 0
4 0 0 30
DEPOT_SECTION
1
EOF"""
        
        # Write to temp file
        mcvrp_file = tmp_path / "test-mcvrp-3.dat"
        mcvrp_file.write_text(mcvrp_content)
        
        # Parse instance
        instance = parse_mcvrp(str(mcvrp_file))
        
        # Verify parsing
        assert instance.name == "test-mcvrp-3"
        assert instance.dimension == 4
        assert instance.vehicles == 2
        
        # Verify demands - each customer wants different product
        assert instance.demands[2] == (30, 0, 0)  # Customer 2 wants product 1
        assert instance.demands[3] == (0, 30, 0)  # Customer 3 wants product 2
        assert instance.demands[4] == (0, 0, 30)  # Customer 4 wants product 3
        
        # Convert to FSM format
        customers_df, params = convert_mcvrp_to_fsm(
            instance_name="test-mcvrp-3",
            custom_instance_path=mcvrp_file
        )
        
        # Verify conversion
        assert len(customers_df) == 3
        
        # Check multiple product types exist
        demand_cols = [col for col in customers_df.columns if col.endswith('_Demand')]
        assert len(demand_cols) >= 3
        
        # Override parameters for quick test
        params.time_limit_minutes = 0.1  # 6 seconds
        params.results_dir = tmp_path
        
        # Run optimization
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )
        
        # Verify solution
        assert solution is not None
        assert solution.solver_status is not None
        
        # Check basic solution properties
        assert solution.solver_status in ['Optimal', 'Feasible', 'TimeLimit']
        
        # Verify cost components exist
        assert solution.total_fixed_cost is not None
        assert solution.total_variable_cost is not None
        
        # For MCVRP, should use vehicles
        if isinstance(solution.vehicles_used, dict):
            total_vehicles = sum(solution.vehicles_used.values())
            assert total_vehicles > 0

    def test_cvrp_vrp_mode(self, tmp_path):
        """Test CVRP solving in VRP mode to verify route sequences."""
        # Create slightly larger instance
        cvrp_content = """NAME: vrp-test
COMMENT: Test VRP mode
TYPE: CVRP
DIMENSION: 6
EDGE_WEIGHT_TYPE: EUC_2D
CAPACITY: 50
NODE_COORD_SECTION
1 0 0
2 2 0
3 0 2
4 -2 0
5 0 -2
6 1 1
DEMAND_SECTION
1 0
2 20
3 20
4 20
5 20
6 10
DEPOT_SECTION
1
-1
EOF"""
        
        cvrp_file = tmp_path / "vrp-test.vrp"
        cvrp_file.write_text(cvrp_content)
        
        # Use unified interface
        customers_df, params = convert_to_fsm(
            VRPType.CVRP,
            instance_names=["vrp-test"],
            benchmark_type=CVRPBenchmarkType.NORMAL,
            custom_instance_paths={"vrp-test": cvrp_file}
        )
        
        # Enable VRP mode
        params.use_vrp_solver = True
        params.time_limit_minutes = 0.2  # 12 seconds
        params.results_dir = tmp_path
        
        # Run optimization
        solution, configs_df = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )
        
        # Verify VRP-specific output
        assert solution is not None
        
        assert not solution.selected_clusters.empty

    def test_benchmark_type_split(self, tmp_path):
        """Test CVRP with SPLIT benchmark type."""
        # Create instance
        cvrp_content = """NAME: split-test
COMMENT: Test split demands
TYPE: CVRP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
CAPACITY: 100
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
DEMAND_SECTION
1 0
2 60
3 60
DEPOT_SECTION
1
-1
EOF"""
        
        cvrp_file = tmp_path / "split-test.vrp"
        cvrp_file.write_text(cvrp_content)
        
        # Convert with SPLIT type - demands split across products
        customers_df, params = convert_cvrp_to_fsm(
            instance_names=["split-test"],
            benchmark_type=CVRPBenchmarkType.SPLIT,
            num_goods=3,
            custom_instance_paths={"split-test": cvrp_file}
        )
        
        # Verify demands are split
        demand_cols = [col for col in customers_df.columns if col.endswith('_Demand')]
        assert len(demand_cols) == 3
        
        # Each customer's demand should be split across products
        for _, customer in customers_df.iterrows():
            total_demand = sum(customer[col] for col in demand_cols)
            assert total_demand > 0  # Should have some demand
            
            # In SPLIT mode, demand should be distributed
            non_zero_demands = sum(1 for col in demand_cols if customer[col] > 0)
            assert non_zero_demands >= 1  # At least one product has demand
        
        # Quick solve
        params.time_limit_minutes = 0.1
        params.results_dir = tmp_path
        
        solution, _ = run_optimization(
            customers_df=customers_df,
            params=params,
            verbose=False
        )
        
        assert solution is not None
        assert solution.solver_status in ['Optimal', 'Feasible', 'TimeLimit']

    def test_small_real_dataset(self):
        """Test with real small dataset if available."""
        # Check for small MCVRP instance
        dataset_path = Path(__file__).parent.parent.parent / "src/fleetmix/benchmarking/datasets/mcvrp/10_3_3_1_(01).dat"
        
        if not dataset_path.exists():
            pytest.skip("Real dataset not found")
            
        # Parse real instance
        instance = parse_mcvrp(str(dataset_path))
        
        # Should be 10 customers + depot
        assert instance.dimension == 11
        
        # Convert to FSM
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            customers_df, params = convert_mcvrp_to_fsm(
                instance_name="10_3_3_1_(01)"
            )
            
            # Should have 10 customers
            assert len(customers_df) == 10
            
            # Override for very quick test
            params.time_limit_minutes = 0.05  # 3 seconds
            params.results_dir = tmp_path
            
            # Just verify it runs without error
            solution, _ = run_optimization(
                customers_df=customers_df,
                params=params,
                verbose=False
            )
            
            assert solution is not None 