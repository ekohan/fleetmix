"""Unit tests for the save_results module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
                customers=["C", "D"],
                total_demand={"Dry": 200, "Chilled": 0, "Frozen": 100},
                centroid_latitude=4.6,
                centroid_longitude=-74.1,
                goods_in_config=["Dry", "Frozen"],
                route_time=4.2,
                method="hierarchical",
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
        )

        self.configurations = [
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

        runtime_params = RuntimeParams()

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
                configurations=self.configurations,
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
                configurations=self.configurations,
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
            configurations=self.configurations,
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
        )

        save_optimization_results(
            solution=solution_with_times,
            configurations=self.configurations,
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

if __name__ == "__main__":
    unittest.main()
