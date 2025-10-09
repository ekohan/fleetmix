"""Unit tests for the save_results module."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import pandas as pd

from fleetmix.config import load_fleetmix_params
from fleetmix.config.params import FleetmixParams, IOParams, AlgorithmParams, ProblemParams, RuntimeParams
from fleetmix.core_types import (
    BenchmarkType,
    Cluster,
    DepotLocation,
    FleetmixSolution,
    VehicleConfiguration,
    VehicleSpec,
    VRPSolution,
)
from fleetmix.utils.save_results import (
    _write_to_excel,
    _write_to_json,
    save_optimization_results,
    visualize_clusters,
)


class TestSaveOptimizationResults(unittest.TestCase):
    """Test cases for save_optimization_results function."""

    def setUp(self):
        """Set up test data."""
        # Create Cluster objects for testing
        clusters = [
            Cluster(
                cluster_id="CL1",
                config_id="C1",
                vehicle_type="Test",
                customers=["A", "B"],
                total_demand={"Dry": 100, "Chilled": 50, "Frozen": 0},
                centroid_latitude=4.5,
                centroid_longitude=-74.0,
                goods_in_config=["Dry", "Chilled"],
                route_time=3.5,
                method="kmeans",
            ),
            Cluster(
                cluster_id="CL2",
                config_id="C2",
                vehicle_type="Test",
                customers=["C", "D"],
                total_demand={"Dry": 200, "Chilled": 0, "Frozen": 100},
                centroid_latitude=4.6,
                centroid_longitude=-74.1,
                goods_in_config=["Dry", "Frozen"],
                route_time=4.2,
                method="hierarchical",
            ),
        ]

        configs = [
            VehicleConfiguration(
                config_id="C1",
                vehicle_type="Type1",
                capacity=1000,
                fixed_cost=100,
                compartments={"Dry": True, "Chilled": True, "Frozen": False},
            ),
            VehicleConfiguration(
                config_id="C2",
                vehicle_type="Type2",
                capacity=2000,
                fixed_cost=200,
                compartments={"Dry": True, "Frozen": True, "Chilled": False},
            ),
        ]
        
        # Create a FleetmixSolution instance for testing
        self.solution = FleetmixSolution(
            selected_clusters=clusters,
            total_fixed_cost=300.0,
            total_variable_cost=150.0,
            total_light_load_penalties=20.0,
            total_compartment_penalties=10.0,
            total_penalties=30.0,
            total_cost=480.0,
            vehicles_used=pd.Series({"Type1": 1, "Type2": 1}).to_dict(),
            total_vehicles=2,
            missing_customers=set(),
            solver_name="TestSolver",
            solver_status="Optimal",
            solver_runtime_sec=10.5,
            time_measurements=None,
            configurations=configs,
        )



        # Create a real FleetmixParams object instead of a mock
        problem_params = ProblemParams(
            vehicles={
                "Type1": VehicleSpec(
                    capacity=1000,
                    fixed_cost=100,
                    compartments={"Dry": True, "Chilled": True, "Frozen": False},
                    extra={},
                ),
                "Type2": VehicleSpec(
                    capacity=2000,
                    fixed_cost=200,
                    compartments={"Dry": True, "Frozen": True, "Chilled": False},
                    extra={},
                ),
            },
            depot=DepotLocation(latitude=4.5, longitude=-74.0),
            goods=["Dry", "Chilled", "Frozen"],
            variable_cost_per_hour=50.0,
            light_load_penalty=10.0,
            light_load_threshold=0.5,
            compartment_setup_cost=5.0,
        )

        algorithm_params = AlgorithmParams(
            clustering_max_depth=3,
            clustering_method="kmeans",
            clustering_distance="euclidean",
            geo_weight=0.5,
            demand_weight=0.5,
            route_time_estimation="BHH",
        )

        io_params = IOParams(
            demand_file="test_demand.csv",
            results_dir=Path(tempfile.gettempdir()),
            format="xlsx",
        )

        runtime_params = RuntimeParams(config="test_config.yaml")

        self.parameters = FleetmixParams(
            problem=problem_params,
            algorithm=algorithm_params,
            io=io_params,
            runtime=runtime_params,
        )

    @patch("fleetmix.utils.save_results._write_to_excel")
    @patch("fleetmix.utils.save_results.visualize_clusters")
    def test_save_optimization_results_excel(self, mock_visualize, mock_write_excel):
        """Test saving optimization results to Excel."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_file = f.name

        try:
            save_optimization_results(
                solution=self.solution,
                parameters=self.parameters,
                filename=temp_file,
                format="xlsx",
            )

            # Check that write function was called
            mock_write_excel.assert_called_once()

            # Check that visualization was created
            mock_visualize.assert_called_once()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("fleetmix.utils.save_results._write_to_json")
    @patch("fleetmix.utils.save_results.visualize_clusters")
    def test_save_optimization_results_json(self, mock_visualize, mock_write_json):
        """Test saving optimization results to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            save_optimization_results(
                solution=self.solution,
                parameters=self.parameters,
                filename=temp_file,
                format="json",
            )

            # Check that write function was called
            mock_write_json.assert_called_once()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("fleetmix.utils.save_results.datetime")
    @patch("fleetmix.utils.save_results._write_to_excel")
    @patch("fleetmix.utils.save_results.visualize_clusters")
    def test_save_optimization_results_default_filename(
        self, mock_visualize, mock_write_excel, mock_datetime
    ):
        """Test saving with auto-generated filename."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        save_optimization_results(
            solution=self.solution,
            parameters=self.parameters,
            format="xlsx",  # Explicitly specify xlsx format to ensure _write_to_excel is called
        )

        # Check that filename was generated with timestamp
        call_args = mock_write_excel.call_args[0]
        filename = call_args[0]
        self.assertIn("optimization_results_20240101_120000", str(filename))

    @patch("fleetmix.utils.save_results._write_to_excel")
    @patch("fleetmix.utils.save_results.visualize_clusters")
    def test_save_optimization_results_with_time_measurements(
        self, mock_visualize, mock_write_excel
    ):
        """Test saving with time measurements."""
        from fleetmix.utils.time_measurement import TimeMeasurement

        time_measurements_list = [
            TimeMeasurement("step1", 1.0, 0.5, 0.1, 0.2, 0.1),
            TimeMeasurement("step2", 2.0, 1.0, 0.2, 0.3, 0.2),
        ]

        # Create a solution object that includes these time_measurements
        solution_with_times = FleetmixSolution(
            selected_clusters=self.solution.selected_clusters,
            total_fixed_cost=self.solution.total_fixed_cost,
            total_variable_cost=self.solution.total_variable_cost,
            total_light_load_penalties=self.solution.total_light_load_penalties,
            total_compartment_penalties=self.solution.total_compartment_penalties,
            total_penalties=self.solution.total_penalties,
            total_cost=self.solution.total_cost,
            vehicles_used=self.solution.vehicles_used,
            total_vehicles=self.solution.total_vehicles,
            missing_customers=self.solution.missing_customers,
            solver_name=self.solution.solver_name,
            solver_status=self.solution.solver_status,
            solver_runtime_sec=self.solution.solver_runtime_sec,
            time_measurements=time_measurements_list,
            configurations=self.solution.configurations,
        )

        save_optimization_results(
            solution=solution_with_times,
            parameters=self.parameters,
            format="xlsx",  # Explicitly specify xlsx format to ensure _write_to_excel is called
        )

        # Check that write function was called
        mock_write_excel.assert_called_once()
        
        # Check that time measurements were included in the call
        call_args = mock_write_excel.call_args[0]
        data = call_args[1]
        self.assertIn("time_measurements_excel", data)
        self.assertEqual(
            len(data["time_measurements_excel"]), 12
        )  # 6 metrics per measurement * 2 measurements


class TestWriteToExcel(unittest.TestCase):
    """Test cases for _write_to_excel function."""

    def setUp(self):
        """Set up test data."""
        self.data = {
            "summary_metrics": [("Metric1", "Value1"), ("Metric2", "Value2")],
            "configurations_df": pd.DataFrame({"Config": ["A", "B"]}),
            "cluster_details": pd.DataFrame(
                {
                    "Cluster_ID": ["C1", "C2"],
                    "Customers": ["[A, B]", "[C, D]"],
                    "TSP_Sequence": ["A -> B", "C -> D"],
                }
            ),
            "vehicles_used": {"Type1": 2, "Type2": 3},
            "other_considerations": [("Consider1", "Value1")],
            "execution_details": [("Detail1", "Value1")],
        }

    def test_write_to_excel_basic(self):
        """Test basic Excel writing functionality."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_file = f.name

        try:
            _write_to_excel(temp_file, self.data)
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_excel_with_time_measurements(self):
        """Test Excel writing with time measurements."""
        self.data["time_measurements_excel"] = [("time1", 1.0), ("time2", 2.0)]

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_file = f.name

        try:
            _write_to_excel(temp_file, self.data)
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestWriteToJSON(unittest.TestCase):
    """Test cases for _write_to_json function."""

    def setUp(self):
        """Set up test data."""
        self.data = {
            "summary_metrics": [("Metric1", "Value1"), ("Metric2", "Value2")],
            "configurations_df": pd.DataFrame({"Config": ["A", "B"]}),
            "cluster_details": pd.DataFrame(
                {"Cluster_ID": ["C1", "C2"], "TSP_Sequence": [["A", "B"], ["C", "D"]]}
            ),
            "vehicles_used": {"Type1": 2, "Type2": 3},
            "other_considerations": [("Consider1", "Value1")],
            "execution_details": [("Detail1", "Value1")],
        }

    def test_write_to_json_basic(self):
        """Test basic JSON writing functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            _write_to_json(temp_file, self.data)

            # Read and verify JSON
            with open(temp_file) as f:
                json_data = json.load(f)

            self.assertIn("Solution Summary", json_data)
            self.assertIn("Configurations", json_data)
            self.assertIn("Selected Clusters", json_data)
            self.assertIn("Vehicle Usage", json_data)

        finally:
            os.unlink(temp_file)

    def test_write_to_json_numpy_types(self):
        """Test JSON writing with numpy types."""
        import numpy as np

        self.data["cluster_details"] = pd.DataFrame(
            {"Cluster_ID": ["C1"], "Value": [np.int64(42)], "Float": [np.float64(3.14)]}
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            _write_to_json(temp_file, self.data)

            # Should not raise JSON serialization error
            with open(temp_file) as f:
                json_data = json.load(f)

            # Check that numpy types were converted
            cluster = json_data["Selected Clusters"][0]
            self.assertEqual(cluster["Value"], 42)
            self.assertAlmostEqual(cluster["Float"], 3.14)

        finally:
            os.unlink(temp_file)


class TestVisualizeCluster(unittest.TestCase):
    """Test cases for visualize_clusters function."""

    def setUp(self):
        """Set up test data."""
        self.selected_clusters = pd.DataFrame(
            {
                "Cluster_ID": ["CL1", "CL2"],
                "Config_ID": ["C1", "C2"],
                "Method": ["kmeans", "hierarchical"],
                "Centroid_Latitude": [4.5, 4.6],
                "Centroid_Longitude": [-74.0, -74.1],
                "Total_Demand": [
                    {"Dry": 100, "Chilled": 50},
                    {"Dry": 200, "Frozen": 100},
                ],
                "Customers": [["A", "B"], ["C", "D"]],
                "Route_Time": [3.5, 4.2],
            }
        )
        self.depot_coords = (4.4, -73.9)

    @patch("folium.Map")
    def test_visualize_clusters_basic(self, mock_map):
        """Test basic cluster visualization."""
        mock_map_instance = MagicMock()
        mock_map.return_value = mock_map_instance

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            base_filename = f.name

        try:
            visualize_clusters(self.selected_clusters, self.depot_coords, base_filename)

            # Check map was created
            mock_map.assert_called_once()

            # Check save was called with correct filename
            expected_filename = base_filename.rsplit(".", 1)[0] + "_clusters.html"
            mock_map_instance.save.assert_called_once_with(expected_filename)

        finally:
            if os.path.exists(base_filename):
                os.unlink(base_filename)

    @patch("folium.Map")
    @patch("folium.Marker")
    @patch("folium.CircleMarker")
    def test_visualize_clusters_markers(self, mock_circle, mock_marker, mock_map):
        """Test that markers are created for depot and clusters."""
        mock_map_instance = MagicMock()
        mock_map.return_value = mock_map_instance

        visualize_clusters(self.selected_clusters, self.depot_coords, "test.xlsx")

        # Check depot marker was created
        mock_marker.assert_called_once()
        depot_call = mock_marker.call_args
        self.assertEqual(depot_call[1]["location"], self.depot_coords)

        # Check cluster markers were created (2 clusters)
        self.assertEqual(mock_circle.call_count, 2)


class TestSaveWithSplitStops(unittest.TestCase):
    """Test cases for save_results with split-stop customers."""

    def setUp(self):
        """Set up test data."""
        configs = [
            VehicleConfiguration(
                config_id="C1",
                vehicle_type="Type1",
                capacity=1000,
                fixed_cost=100,
                compartments={"Dry": True, "Chilled": True, "Frozen": False},
            ),
            VehicleConfiguration(
                config_id="C2",
                vehicle_type="Type2",
                capacity=1500,
                fixed_cost=180,
                compartments={"Dry": True, "Chilled": False, "Frozen": True},
            ),
        ]
        
        problem_params = ProblemParams(
            vehicles={
                "Type1": VehicleSpec(
                    capacity=1000,
                    fixed_cost=100,
                    compartments={"Dry": True, "Chilled": True, "Frozen": False},
                    extra={},
                ),
                "Type2": VehicleSpec(
                    capacity=1500,
                    fixed_cost=180,
                    compartments={"Dry": True, "Chilled": False, "Frozen": True},
                    extra={},
                ),
            },
            depot=DepotLocation(latitude=4.5, longitude=-74.0),
            goods=["Dry", "Chilled", "Frozen"],
            variable_cost_per_hour=50.0,
            light_load_penalty=10.0,
            light_load_threshold=0.5,
            compartment_setup_cost=5.0,
        )

        algorithm_params = AlgorithmParams(
            clustering_max_depth=3,
            clustering_method="kmeans",
        )

        io_params = IOParams(
            demand_file="test_demand.csv",
            results_dir=Path(tempfile.gettempdir()),
            format="xlsx",
        )

        runtime_params = RuntimeParams(config="test_config.yaml")

        self.parameters = FleetmixParams(
            problem=problem_params,
            algorithm=algorithm_params,
            io=io_params,
            runtime=runtime_params,
        )
        self.configs = configs

    @patch("fleetmix.utils.save_results.visualize_clusters")
    def test_save_with_split_stop_customers(self, mock_visualize):
        """Test saving results with split-stop customers (lines 86-98)."""
        # Create clusters with split-stop customer notation (using ::)
        clusters_with_split_stops = [
            Cluster(
                cluster_id="CL1",
                config_id="C1",
                vehicle_type="Test",
                customers=["A::Dry", "B::Chilled"],  # Split-stop notation
                total_demand={"Dry": 100, "Chilled": 50, "Frozen": 0},
                centroid_latitude=4.5,
                centroid_longitude=-74.0,
                goods_in_config=["Dry", "Chilled"],
                route_time=3.5,
                method="kmeans",
            ),
            Cluster(
                cluster_id="CL2",
                config_id="C2",
                vehicle_type="Test",
                customers="C",  # Single customer as string (line 88-91)
                total_demand={"Dry": 200, "Chilled": 0, "Frozen": 100},
                centroid_latitude=4.6,
                centroid_longitude=-74.1,
                goods_in_config=["Dry", "Frozen"],
                route_time=4.2,
                method="hierarchical",
            ),
        ]

        solution = FleetmixSolution(
            selected_clusters=clusters_with_split_stops,
            total_cost=480.0,
            configurations=self.configs,
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_file = f.name

        try:
            save_optimization_results(
                solution=solution,
                parameters=self.parameters,
                filename=temp_file,
                format="xlsx",
            )
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
            mock_visualize.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("fleetmix.utils.save_results.visualize_clusters")
    @patch("fleetmix.utils.save_results._write_to_excel")
    def test_split_stop_metrics_deduplicated(self, mock_write_excel, mock_visualize):
        """Ensure allow_split_stops deduplicates customers and computes load metrics."""
        parameters = dataclasses.replace(
            self.parameters,
            problem=dataclasses.replace(
                self.parameters.problem, allow_split_stops=True
            ),
        )

        clusters_with_split_stops = [
            Cluster(
                cluster_id="CL1",
                config_id="C1",
                vehicle_type="Test",
                customers=["A::Dry", "A::Frozen", "B::Chilled"],
                total_demand={"Dry": 100, "Chilled": 50, "Frozen": 25},
                centroid_latitude=4.5,
                centroid_longitude=-74.0,
                goods_in_config=["Dry", "Chilled", "Frozen"],
                route_time=3.5,
                method="kmeans",
            ),
            Cluster(
                cluster_id="CL2",
                config_id="C2",
                vehicle_type="Test",
                customers="C::Dry",
                total_demand={"Dry": 300, "Chilled": 0, "Frozen": 0},
                centroid_latitude=4.6,
                centroid_longitude=-74.1,
                goods_in_config=["Dry"],
                route_time=4.2,
                method="hierarchical",
            ),
        ]

        solution = FleetmixSolution(
            selected_clusters=clusters_with_split_stops,
            configurations=self.configs,
            vehicles_used={"Type1": 1, "Type2": 1},
            total_vehicles=2,
        )

        save_optimization_results(
            solution=solution,
            parameters=parameters,
            filename="dummy.xlsx",
            format="xlsx",
        )

        mock_write_excel.assert_called_once()
        mock_visualize.assert_called_once()
        _, data = mock_write_excel.call_args[0]
        cluster_details = data["cluster_details"]

        cl1 = cluster_details.loc[cluster_details["Cluster_ID"] == "CL1"].iloc[0]
        cl2 = cluster_details.loc[cluster_details["Cluster_ID"] == "CL2"].iloc[0]

        self.assertEqual(cl1["Customers"], "['A', 'B']")
        self.assertEqual(cl1["Num_Customers"], 2)
        self.assertEqual(cl2["Customers"], "['C']")
        self.assertEqual(cl2["Num_Customers"], 1)

        self.assertAlmostEqual(cl1["Demand_Dry_pct"], 100 / 175)
        self.assertAlmostEqual(cl1["Demand_Chilled_pct"], 50 / 175)
        self.assertAlmostEqual(cl1["Demand_Frozen_pct"], 25 / 175)
        self.assertAlmostEqual(cl1["Load_Dry_pct"], 100 / 1000)
        self.assertAlmostEqual(cl1["Load_Chilled_pct"], 50 / 1000)
        self.assertAlmostEqual(cl1["Load_Frozen_pct"], 25 / 1000)
        self.assertAlmostEqual(cl1["Load_total_pct"], 175 / 1000)
        self.assertAlmostEqual(cl1["Load_empty_pct"], 1 - (175 / 1000))

        self.assertAlmostEqual(cl2["Demand_Dry_pct"], 1.0)
        self.assertAlmostEqual(cl2["Load_Dry_pct"], 300 / 1500)
        self.assertAlmostEqual(cl2["Load_total_pct"], 300 / 1500)

        summary = dict(data["summary_metrics"])
        self.assertEqual(summary["Customers per Cluster (Max)"], "2")
        self.assertEqual(summary["Customers per Cluster (Min)"], "1")

        # Personal DepotLocation access to cover __getitem__ branch via params
        depot = self.parameters.problem.depot
        self.assertIsInstance(depot["latitude"], float)
        self.assertIsInstance(depot["longitude"], float)


@patch("fleetmix.utils.save_results.visualize_clusters")
class TestSaveResultsJson(unittest.TestCase):
    def setUp(self):
        self.params = FleetmixParams(
            problem=ProblemParams(
                vehicles={
                    "Truck": VehicleSpec(
                        capacity=1000,
                        fixed_cost=100,
                        compartments={"Dry": True},
                    )
                },
                depot=DepotLocation(latitude=4.5, longitude=-74.0),
                goods=["Dry"],
                variable_cost_per_hour=50.0,
            ),
            algorithm=AlgorithmParams(),
            io=IOParams(
                demand_file="test_demand.csv",
                results_dir=Path(tempfile.gettempdir()),
                format="json",
            ),
            runtime=RuntimeParams(config="test_config.yaml"),
        )

        self.solution = FleetmixSolution(
            selected_clusters=[
                Cluster(
                    cluster_id="CL1",
                    config_id="1",
                    vehicle_type="Truck",
                    customers=["A", "B"],
                    total_demand={"Dry": 100},
                    centroid_latitude=4.5,
                    centroid_longitude=-74.0,
                    goods_in_config=["Dry"],
                    route_time=3.0,
                    method="kmeans",
                    tsp_sequence=["Depot", "A", "B", "Depot"],
                )
            ],
            total_fixed_cost=100.0,
            total_variable_cost=50.0,
            total_penalties=0.0,
            total_cost=150.0,
            vehicles_used={"Truck": 1},
            total_vehicles=1,
            configurations=[
                VehicleConfiguration(
                    config_id="1",
                    vehicle_type="Truck",
                    capacity=1000,
                    fixed_cost=100,
                    compartments={"Dry": True},
                )
            ],
        )

    def test_json_writer_serializes_numpy(self, mock_visualize):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            save_optimization_results(
                solution=self.solution,
                parameters=self.params,
                filename=str(temp_file),
                format="json",
            )

            data = json.loads(temp_file.read_text())
            assert "Solution Summary" in data
            clusters = data["Selected Clusters"]
            assert clusters[0]["TSP_Sequence"].startswith("Depot")
        finally:
            temp_file.unlink(missing_ok=True)

    @patch("fleetmix.utils.save_results._write_to_json", side_effect=ValueError("boom"))
    def test_json_write_error_surface(self, mock_json, mock_visualize):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        with self.assertRaises(ValueError):
            save_optimization_results(
                solution=self.solution,
                parameters=self.params,
                filename=str(temp_file),
                format="json",
            )

        mock_json.assert_called_once()

    def test_json_includes_expected_vehicles(self, mock_visualize):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        expected = 3

        save_optimization_results(
            solution=self.solution,
            parameters=self.params,
            filename=str(temp_file),
            format="json",
            expected_vehicles=expected,
        )

        data = json.loads(temp_file.read_text())
        summary = data["Solution Summary"]
        assert summary.get("Expected Vehicles") == expected


if __name__ == "__main__":
    unittest.main()
