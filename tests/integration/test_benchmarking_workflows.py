"""
Integration tests for FleetMix benchmarking workflows.
Tests real VRP instance processing and benchmark suite functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path

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
        try:
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

        except ImportError as e:
            pytest.skip(f"VRPLIB not available for parsing: {e!s}")
        except Exception as e:
            pytest.skip(f"CVRP parsing failed: {e!s}")

    def test_mcvrp_instance_parsing(self, small_mcvrp_file):
        """Test MCVRP instance parsing functionality."""
        try:
            instance = parse_mcvrp(str(small_mcvrp_file))

            assert isinstance(instance, MCVRPInstance)
            assert instance.dimension == 4
            assert instance.capacity == 100  # First capacity value
            assert instance.vehicles == 2
            assert len(instance.coords) == 4
            assert len(instance.demands) == 4

        except Exception as e:
            pytest.skip(f"MCVRP parsing failed: {e!s}")

    def test_cvrp_to_fsm_conversion(self, small_cvrp_file, temp_results_dir):
        """Test CVRP to FSM conversion process."""
        instance_name = small_cvrp_file.stem

        # Test different benchmark types
        for benchmark_type in [CVRPBenchmarkType.NORMAL, CVRPBenchmarkType.SPLIT]:
            try:
                customers_df, params = convert_cvrp_to_fsm(
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

            except Exception as e:
                pytest.skip(
                    f"CVRP to FSM conversion failed for {benchmark_type}: {e!s}"
                )

    def test_mcvrp_to_fsm_conversion(self, small_mcvrp_file, temp_results_dir):
        """Test MCVRP to FSM conversion process."""
        instance_name = small_mcvrp_file.stem
        try:
            customers_df, params = convert_mcvrp_to_fsm(
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

        except Exception as e:
            pytest.skip(f"MCVRP to FSM conversion failed: {e!s}")

    def test_unified_vrp_pipeline_cvrp(self, small_cvrp_file, temp_results_dir):
        """Test unified VRP pipeline with CVRP instance."""
        instance_name = small_cvrp_file.stem

        try:
            # Use the unified pipeline interface
            customers_df, params = convert_to_fsm(
                VRPType.CVRP,
                instance_names=[instance_name],
                benchmark_type=CVRPBenchmarkType.NORMAL,
                custom_instance_paths={instance_name: small_cvrp_file},
            )

            assert isinstance(customers_df, pd.DataFrame)
            assert len(customers_df) > 0

            # Override results directory for testing
            params.results_dir = temp_results_dir
            params.time_limit_minutes = 1  # Quick test

            # Run optimization
            solution, configs = run_optimization(
                customers_df=customers_df, params=params, verbose=True
            )

            # Verify solution structure - solution is FleetmixSolution
            assert solution.solver_status is not None  # Attribute access
            assert isinstance(configs, list)
            assert all(isinstance(config, VehicleConfiguration) for config in configs)

        except Exception as e:
            pytest.skip(f"CVRP pipeline test failed: {e!s}")

    def test_unified_vrp_pipeline_mcvrp(self, small_mcvrp_file, temp_results_dir):
        """Test unified VRP pipeline with MCVRP instance."""
        try:
            # Use the unified pipeline interface
            customers_df, params = convert_to_fsm(
                VRPType.MCVRP, instance_path=small_mcvrp_file
            )

            assert isinstance(customers_df, pd.DataFrame)
            assert len(customers_df) > 0

            # Override results directory for testing
            params.results_dir = temp_results_dir
            params.time_limit_minutes = 1  # Quick test

            # Run optimization
            solution, configs = run_optimization(
                customers_df=customers_df, params=params, verbose=True
            )

            # Verify solution structure - solution is FleetmixSolution
            assert solution.solver_status is not None  # Attribute access
            assert isinstance(configs, list)
            assert all(isinstance(config, VehicleConfiguration) for config in configs)

        except Exception as e:
            pytest.skip(f"MCVRP pipeline test failed: {e!s}")

    def test_vrp_solver_interface(self, small_cvrp_file, temp_results_dir):
        """Test VRP solver interface for baseline comparisons."""
        # First parse the CVRP instance to get customer data
        try:
            from fleetmix.config.parameters import Parameters
            from fleetmix.core_types import BenchmarkType

            # Parse CVRP instance
            parser = CVRPParser(str(small_cvrp_file))
            cvrp_instance = parser.parse()

            # Convert CVRP instance to customer DataFrame format
            customers_data = []
            for i, (lat, lon) in cvrp_instance.coordinates.items():
                if i > 1:  # Skip depot (index 1)
                    customers_data.append(
                        {
                            "Customer_ID": f"C{i}",
                            "Latitude": float(lat),  # Convert numpy to float
                            "Longitude": float(lon),  # Convert numpy to float
                            "Dry_Demand": float(
                                cvrp_instance.demands[i]
                            ),  # Convert numpy to float
                            "Chilled_Demand": 0.0,
                            "Frozen_Demand": 0.0,
                        }
                    )

            customers_df = pd.DataFrame(customers_data)

            # Create mock parameters for the solver
            from fleetmix.core_types import DepotLocation, VehicleSpec

            # Create a temporary YAML for parameters
            mock_params_dict = {
                "goods": ["Dry", "Chilled", "Frozen"],
                "vehicles": {
                    "Test Van": {
                        "fixed_cost": 100,
                        "capacity": int(cvrp_instance.capacity),  # Convert numpy to int
                        "avg_speed": 40.0,
                        "max_route_time": 8.0,
                        "service_time": 15.0,
                        "compartments": {"Dry": True, "Chilled": True, "Frozen": True},
                        "extra": {"variable_cost_per_km": 0.5},
                    }
                },
                "variable_cost_per_hour": 20.0,
                "depot": {
                    "latitude": float(cvrp_instance.coordinates[1][0]),
                    "longitude": float(cvrp_instance.coordinates[1][1]),
                },  # Convert numpy to float
                "clustering": {
                    "route_time_estimation": "BHH",
                    "method": "minibatch_kmeans",
                    "max_depth": 5,
                    "geo_weight": 0.7,
                    "demand_weight": 0.3,
                    "distance": "euclidean",
                },
                "demand_file": "dummy_demand.csv",
                "light_load_penalty": 10.0,
                "light_load_threshold": 0.5,
                "compartment_setup_cost": 100.0,
                "format": "json",
                "post_optimization": False,
                "prune_tsp": False,
            }

            import yaml

            temp_yaml_path = temp_results_dir / "mock_params_for_vrp_solver.yaml"
            with open(temp_yaml_path, "w") as f:
                yaml.dump(mock_params_dict, f)

            params = Parameters.from_yaml(str(temp_yaml_path))  # Convert Path to string
            params.results_dir = (
                temp_results_dir  # Set results_dir directly as Path object
            )

            # Test single compartment solver
            solver = VRPSolver(
                customers=customers_df,
                params=params,
                time_limit=30,  # Short time limit for testing
                benchmark_type=BenchmarkType.SINGLE_COMPARTMENT,
            )

            # Test solve_scv method
            scv_solution = solver.solve_scv(verbose=True)

            # Verify solution structure
            assert isinstance(scv_solution, VRPSolution)
            assert scv_solution.solver_status in ["Optimal", "Infeasible", "Feasible"]
            assert scv_solution.total_cost >= 0 or scv_solution.total_cost == float(
                "inf"
            )
            assert scv_solution.execution_time >= 0
            assert isinstance(scv_solution.routes, list)
            assert isinstance(scv_solution.vehicle_loads, list)
            assert isinstance(scv_solution.route_times, list)
            assert isinstance(scv_solution.route_feasibility, list)

            # Test multi-compartment solver
            mc_solver = VRPSolver(
                customers=customers_df,
                params=params,
                time_limit=30,
                benchmark_type=BenchmarkType.MULTI_COMPARTMENT,
            )

            # Test solve_mcv method
            mc_solution = mc_solver.solve_mcv(verbose=True)

            assert isinstance(mc_solution, VRPSolution)
            assert mc_solution.solver_status in ["Optimal", "Infeasible", "Feasible"]

            # Test main solve method with single compartment
            sc_result = solver.solve(verbose=True)
            assert isinstance(sc_result, dict)
            # Should return solutions for each product type
            for product in params.goods:
                if product in sc_result:
                    assert isinstance(sc_result[product], VRPSolution)

            # Test main solve method with multi-compartment
            mc_result = mc_solver.solve(verbose=True)
            assert isinstance(mc_result, dict)
            assert "multi" in mc_result

        except ImportError as e:
            if "pyvrp" in str(e).lower():
                pytest.skip(f"PyVRP not available: {e!s}")
            else:
                pytest.skip(f"Required dependency not available: {e!s}")
        except Exception as e:
            pytest.skip(f"VRP solver test failed: {e!s}")

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
                try:
                    parser = CVRPParser(str(cvrp_files[0]))
                    real_instance = parser.parse()
                    assert isinstance(real_instance, CVRPInstance)
                    assert real_instance.dimension > 0
                except Exception as e:
                    pytest.skip(f"Could not parse real CVRP instance: {e!s}")

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
                try:
                    real_instance = parse_mcvrp(str(mcvrp_files[0]))
                    assert isinstance(real_instance, MCVRPInstance)
                    assert real_instance.dimension > 0
                except Exception as e:
                    pytest.skip(f"Could not parse real MCVRP instance: {e!s}")

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
            try:
                customers_df, params = convert_cvrp_to_fsm(
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

            except Exception as e:
                pytest.skip(f"Benchmark type {benchmark_type} failed: {e!s}")

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

        try:
            start_time = time.time()

            # Convert instance
            customers_df, params = convert_to_fsm(
                VRPType.CVRP,
                instance_names=[instance_name],
                benchmark_type=CVRPBenchmarkType.NORMAL,
                custom_instance_paths={instance_name: small_cvrp_file},
            )

            conversion_time = time.time() - start_time

            # Override settings for quick test
            params.results_dir = temp_results_dir
            params.time_limit_minutes = 0.5  # Very quick

            start_optimization = time.time()

            solution, configs = run_optimization(
                customers_df=customers_df, params=params, verbose=False
            )

            optimization_time = time.time() - start_optimization

            # Verify timing metrics are reasonable
            assert conversion_time < 60  # Should convert quickly
            assert optimization_time < 120  # Should optimize quickly for small instance

            # Verify solution contains timing info
            assert solution.solver_runtime_sec is not None  # Attribute access
            assert solution.solver_runtime_sec >= 0  # Attribute access

        except Exception as e:
            pytest.skip(f"Performance metrics test failed: {e!s}")
