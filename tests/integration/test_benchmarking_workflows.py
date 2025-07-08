"""
Integration tests for FleetMix benchmarking workflows.
Tests real VRP instance processing and benchmark suite functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path
import dataclasses
import pandas as pd
import pytest

from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType, convert_cvrp_to_fsm
from fleetmix.benchmarking.converters.mcvrp import convert_mcvrp_to_fsm
from fleetmix.benchmarking.models.models import CVRPInstance, MCVRPInstance
from fleetmix.benchmarking.parsers.cvrp import CVRPParser
from fleetmix.benchmarking.parsers.mcvrp import parse_mcvrp
from fleetmix.benchmarking.solvers.vrp_solver import VRPSolver
from fleetmix.core_types import FleetmixSolution, VehicleConfiguration, VRPSolution
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.config.params import AlgorithmParams, FleetmixParams, IOParams, RuntimeParams, ProblemParams


class TestBenchmarkingWorkflows:
    """Test benchmarking components with real VRP instances."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def small_cvrp_instance_content(self):
        """Create a small CVRP instance for testing."""
        return """NAME: test-instance
COMMENT: Test CVRP instance
TYPE: CVRP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
CAPACITY: 100
NODE_COORD_SECTION
1 0 0
2 10 10  
3 20 0
4 10 -10
DEMAND_SECTION
1 0
2 30
3 40
4 20
DEPOT_SECTION
1
-1
EOF"""

    @pytest.fixture
    def small_mcvrp_instance_content(self):
        """Create a small MCVRP instance for testing."""
        return """NAME: test-mcvrp-instance
COMMENT: Test MCVRP instance
TYPE: MCVRP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT # Or any other valid type, not strictly parsed by this parser
CAPACITY: 100
VEHICLES: 2
PRODUCT TYPES: 3
COMPARTMENTS: 3
NODE_COORD_SECTION
1 0 0
2 10 10
3 20 0
4 10 -10
DEMAND_SECTION
1 0 0 0
2 30 20 0
3 40 15 0
4 20 25 0
DEPOT_SECTION
1
EOF
"""

    @pytest.fixture
    def small_cvrp_file(self, temp_results_dir, small_cvrp_instance_content):
        """Create a temporary CVRP file."""
        cvrp_file = temp_results_dir / "test-instance.vrp"
        cvrp_file.write_text(small_cvrp_instance_content)
        return cvrp_file

    @pytest.fixture
    def small_mcvrp_file(self, temp_results_dir, small_mcvrp_instance_content):
        """Create a temporary MCVRP file."""
        mcvrp_file = temp_results_dir / "test-instance.dat"
        mcvrp_file.write_text(small_mcvrp_instance_content)
        return mcvrp_file

    def test_cvrp_instance_parsing(self, small_cvrp_file):
        """Test CVRP instance parsing functionality."""
        # Test parsing the CVRP instance
        parser = CVRPParser(str(small_cvrp_file))
        instance = parser.parse()

        assert isinstance(instance, CVRPInstance)
        assert instance.name == "test-instance"
        assert instance.dimension == 4
        assert instance.capacity == 100
        assert len(instance.coordinates) == 4
        assert len(instance.demands) == 4

        # Verify coordinate and demand data
        assert instance.coordinates[1] == (0, 0)  # Depot (1-indexed)
        assert instance.demands[1] == 0  # Depot demand

    def test_mcvrp_instance_parsing(self, small_mcvrp_file):
        """Test MCVRP instance parsing functionality."""
        instance = parse_mcvrp(str(small_mcvrp_file))

        assert isinstance(instance, MCVRPInstance)
        assert instance.dimension == 4
        assert instance.capacity == 100  # First capacity value
        assert instance.vehicles == 2
        assert len(instance.coords) == 4
        assert len(instance.demands) == 4

    def test_cvrp_to_fsm_conversion(self, small_cvrp_file, temp_results_dir):
        """Test CVRP to FSM conversion process."""
        instance_name = small_cvrp_file.stem

        # Test different benchmark types
        for benchmark_type in [CVRPBenchmarkType.NORMAL, CVRPBenchmarkType.SPLIT]:
            customers_df, instance_spec = convert_cvrp_to_fsm(
                instance_names=[instance_name],
                benchmark_type=benchmark_type,
                num_goods=2,
                custom_instance_paths={instance_name: small_cvrp_file},
            )

            # Verify conversion output
            assert isinstance(customers_df, pd.DataFrame)
            assert len(customers_df) == 3  # 3 customers (excluding depot)
            assert "Customer_ID" in customers_df.columns
            assert "Latitude" in customers_df.columns
            assert "Longitude" in customers_df.columns

            # Verify demand columns exist
            demand_cols = ["Dry_Demand", "Chilled_Demand", "Frozen_Demand"]
            demand_present = [
                col for col in demand_cols if col in customers_df.columns
            ]
            assert (
                len(demand_present) >= 2
            )  # At least 2 demand types for num_goods=2

    def test_mcvrp_to_fsm_conversion(self, small_mcvrp_file, temp_results_dir):
        """Test MCVRP to FSM conversion process."""
        instance_name = small_mcvrp_file.stem
        customers_df, instance_spec = convert_mcvrp_to_fsm(
            instance_name=instance_name, custom_instance_path=small_mcvrp_file
        )

        # Verify conversion output
        assert isinstance(customers_df, pd.DataFrame)
        assert len(customers_df) == 3  # 3 customers (excluding depot)
        assert "Customer_ID" in customers_df.columns
        assert "Latitude" in customers_df.columns
        assert "Longitude" in customers_df.columns

        # Verify demand structure for MCVRP
        assert any(col.endswith("_Demand") for col in customers_df.columns)

    def test_unified_vrp_pipeline_cvrp(self, small_cvrp_file, temp_results_dir):
        """Test unified VRP pipeline with CVRP instance."""
        instance_name = small_cvrp_file.stem

        # Use the unified pipeline interface
        customers_df, instance_spec = convert_to_fsm(
            VRPType.CVRP,
            instance_names=[instance_name],
            benchmark_type=CVRPBenchmarkType.NORMAL,
            custom_instance_paths={instance_name: small_cvrp_file},
        )

        assert isinstance(customers_df, pd.DataFrame)
        assert len(customers_df) > 0

        # Convert InstanceSpec to ProblemParams and set up params
        problem_params = ProblemParams(
            expected_vehicles=instance_spec.expected_vehicles,
            depot=instance_spec.depot,
            goods=instance_spec.goods,
            vehicles=instance_spec.vehicles,
            variable_cost_per_hour=50.0
        )
        params = FleetmixParams(
            problem=problem_params,
            algorithm=AlgorithmParams(),
            io=IOParams(
                demand_file=f"{instance_name}.csv",
                results_dir=temp_results_dir,
                format="json",
            ),
            runtime=RuntimeParams(config=Path("test_config.yaml")),
        )

        # Run optimization
        solution, configs = run_optimization(
            customers_df=customers_df, params=params
        )

        # Verify solution structure - solution is FleetmixSolution
        assert solution.solver_status is not None  # Attribute access
        assert isinstance(configs, list)
        assert all(isinstance(config, VehicleConfiguration) for config in configs)

    def test_unified_vrp_pipeline_mcvrp(self, small_mcvrp_file, temp_results_dir):
        """Test unified VRP pipeline with MCVRP instance."""
        # Use the unified pipeline interface
        customers_df, instance_spec = convert_to_fsm(
            VRPType.MCVRP, instance_path=small_mcvrp_file
        )

        assert isinstance(customers_df, pd.DataFrame)
        assert len(customers_df) > 0

        # Convert InstanceSpec to ProblemParams and set up params
        problem_params = ProblemParams(
            expected_vehicles=instance_spec.expected_vehicles,
            depot=instance_spec.depot,
            goods=instance_spec.goods,
            vehicles=instance_spec.vehicles,
            variable_cost_per_hour=50.0
        )
        params = FleetmixParams(
            problem=problem_params,
            algorithm=AlgorithmParams(),
            io=IOParams(
                demand_file=f"{small_mcvrp_file.stem}.dat",
                results_dir=temp_results_dir,
                format="json",
            ),
            runtime=RuntimeParams(config=Path("test_config.yaml")),
        )

        # Run optimization
        solution, configs = run_optimization(
            customers_df=customers_df, params=params
        )

        # Verify solution structure - solution is FleetmixSolution
        assert solution.solver_status is not None  # Attribute access
        assert isinstance(configs, list)
        assert all(isinstance(config, VehicleConfiguration) for config in configs)

    def test_real_dataset_availability(self):
        """Test availability of real benchmark datasets."""
        # Check if real CVRP datasets are available
        cvrp_dir = (
            Path(__file__).parent.parent.parent
            / "src/fleetmix/benchmarking/datasets/cvrp"
        )
        if cvrp_dir.exists():
            cvrp_files = list(cvrp_dir.glob("X-n*.vrp"))
            if cvrp_files:
                print(f"Found {len(cvrp_files)} CVRP instances")

                # Test parsing a real instance
                parser = CVRPParser(str(cvrp_files[0]))
                real_instance = parser.parse()
                assert isinstance(real_instance, CVRPInstance)
                assert real_instance.dimension > 0

        # Check if real MCVRP datasets are available
        mcvrp_dir = (
            Path(__file__).parent.parent.parent
            / "src/fleetmix/benchmarking/datasets/mcvrp"
        )
        if mcvrp_dir.exists():
            mcvrp_files = list(mcvrp_dir.glob("*.dat"))
            if mcvrp_files:
                print(f"Found {len(mcvrp_files)} MCVRP instances")

                # Test parsing a real instance
                real_instance = parse_mcvrp(str(mcvrp_files[0]))
                assert isinstance(real_instance, MCVRPInstance)
                assert real_instance.dimension > 0

    def test_benchmark_type_variations(self, small_cvrp_file):
        """Test different CVRP benchmark type variations."""
        instance_name = small_cvrp_file.stem

        # Test all benchmark types
        benchmark_types = [
            CVRPBenchmarkType.NORMAL,
            CVRPBenchmarkType.SPLIT,
            CVRPBenchmarkType.SCALED,
        ]

        for benchmark_type in benchmark_types:
            customers_df, problem_params = convert_cvrp_to_fsm(
                instance_names=[instance_name],
                benchmark_type=benchmark_type,
                num_goods=3,
                custom_instance_paths={instance_name: small_cvrp_file},
            )

            # Each benchmark type should produce valid output
            assert isinstance(customers_df, pd.DataFrame)
            assert len(customers_df) > 0

            # Verify demands are generated according to benchmark type
            demand_cols = [
                col for col in customers_df.columns if col.endswith("_Demand")
            ]
            assert len(demand_cols) >= 2  # Should have multiple demand types

            # Verify all demands are non-negative
            for col in demand_cols:
                assert all(customers_df[col] >= 0)

    def test_edge_case_empty_instance(self, temp_results_dir):
        """Test handling of malformed/empty instances."""
        # Create malformed CVRP file
        malformed_cvrp = temp_results_dir / "malformed.vrp"
        malformed_cvrp.write_text("INVALID CONTENT")

        # Should handle malformed files gracefully
        with pytest.raises(Exception):  # Should raise some parsing error
            parser = CVRPParser(str(malformed_cvrp))
            parser.parse()

        # Create malformed MCVRP file
        malformed_mcvrp = temp_results_dir / "malformed.dat"
        malformed_mcvrp.write_text("invalid content here")

        with pytest.raises(Exception):  # Should raise some parsing error
            parse_mcvrp(str(malformed_mcvrp))

    def test_benchmark_performance_metrics(self, small_cvrp_file, temp_results_dir):
        """Test that benchmark processing tracks performance metrics."""
        import time

        instance_name = small_cvrp_file.stem

        start_time = time.time()

        # Convert instance
        customers_df, instance_spec = convert_to_fsm(
            VRPType.CVRP,
            instance_names=[instance_name],
            benchmark_type=CVRPBenchmarkType.NORMAL,
            custom_instance_paths={instance_name: small_cvrp_file},
        )

        conversion_time = time.time() - start_time

        # Convert InstanceSpec to ProblemParams and set up params
        problem_params = ProblemParams(
            expected_vehicles=instance_spec.expected_vehicles,
            depot=instance_spec.depot,
            goods=instance_spec.goods,
            vehicles=instance_spec.vehicles,
            variable_cost_per_hour=50.0
        )
        params = FleetmixParams(
            problem=problem_params,
            algorithm=AlgorithmParams(),
            io=IOParams(
                demand_file=f"{instance_name}.csv",
                results_dir=temp_results_dir,
                format="json",
            ),
            runtime=RuntimeParams(config=Path("test_config.yaml")),
        )

        start_optimization = time.time()

        solution, configs = run_optimization(
            customers_df=customers_df, params=params
        )

        optimization_time = time.time() - start_optimization

        # Verify timing metrics are reasonable
        assert conversion_time < 60  # Should convert quickly
        assert optimization_time < 120  # Should optimize quickly for small instance

        # Verify solution contains timing info
        assert solution.solver_runtime_sec is not None  # Attribute access
        assert solution.solver_runtime_sec >= 0  # Attribute access
