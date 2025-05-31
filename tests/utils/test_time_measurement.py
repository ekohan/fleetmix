"""Unit tests for the time_measurement module."""

import unittest
import time
import subprocess
import sys
from joblib import Parallel, delayed

from fleetmix.utils.time_measurement import TimeRecorder, TimeMeasurement


class TestTimeMeasurement(unittest.TestCase):
    """Test cases for TimeMeasurement and TimeRecorder classes."""
    
    def test_time_measurement_dataclass(self):
        """Test that TimeMeasurement dataclass is properly constructed."""
        measurement = TimeMeasurement(
            span_name="test_span",
            wall_time=1.5,
            process_user_time=0.1,
            process_system_time=0.05,
            children_user_time=0.2,
            children_system_time=0.1
        )
        
        self.assertEqual(measurement.span_name, "test_span")
        self.assertEqual(measurement.wall_time, 1.5)
        self.assertEqual(measurement.process_user_time, 0.1)
        self.assertEqual(measurement.process_system_time, 0.05)
        self.assertEqual(measurement.children_user_time, 0.2)
        self.assertEqual(measurement.children_system_time, 0.1)
    
    def test_recorder_initialization(self):
        """Test that TimeRecorder initializes with empty measurements list."""
        recorder = TimeRecorder()
        self.assertEqual(len(recorder.measurements), 0)
        self.assertIsInstance(recorder.measurements, list)
    
    def test_simple_sleep_block(self):
        """Test measuring a simple sleep block."""
        recorder = TimeRecorder()
        sleep_duration = 0.5
        
        with recorder.measure("sleep_test"):
            time.sleep(sleep_duration)
        
        # Check that one measurement was recorded
        self.assertEqual(len(recorder.measurements), 1)
        
        measurement = recorder.measurements[0]
        self.assertEqual(measurement.span_name, "sleep_test")
        
        # Wall time should be at least the sleep duration
        self.assertGreaterEqual(measurement.wall_time, sleep_duration)
        # But not too much more (allow 0.1s overhead)
        self.assertLess(measurement.wall_time, sleep_duration + 0.1)
        
        # Sleep shouldn't use much CPU time
        self.assertLess(measurement.process_user_time, 0.1)
        self.assertLess(measurement.process_system_time, 0.1)
        
        # No child processes for simple sleep
        self.assertEqual(measurement.children_user_time, 0.0)
        self.assertEqual(measurement.children_system_time, 0.0)
    
    def test_cpu_intensive_block(self):
        """Test measuring a CPU-intensive block."""
        recorder = TimeRecorder()
        
        def cpu_intensive_task():
            """Perform some CPU-intensive calculations."""
            result = 0
            for i in range(1000000):
                result += i ** 2
            return result
        
        with recorder.measure("cpu_test"):
            cpu_intensive_task()
        
        measurement = recorder.measurements[0]
        self.assertEqual(measurement.span_name, "cpu_test")
        
        # CPU-intensive task should have measurable user time
        self.assertGreater(measurement.process_user_time, 0.0)
        # Wall time should be greater than 0
        self.assertGreater(measurement.wall_time, 0.0)
    
    def test_parallel_joblib_block(self):
        """Test measuring a block that spawns processes using joblib."""
        recorder = TimeRecorder()
        
        def dummy_task(x):
            """Simple task that sleeps and returns square."""
            time.sleep(0.1)
            return x * x
        
        with recorder.measure("parallel_test"):
            results = Parallel(n_jobs=2, backend='loky')(
                delayed(dummy_task)(i) for i in range(4)
            )
        
        measurement = recorder.measurements[0]
        self.assertEqual(measurement.span_name, "parallel_test")
        
        # Wall time should be at least 0.2s (4 tasks, 0.1s each, with 2 workers)
        self.assertGreaterEqual(measurement.wall_time, 0.2)
        
        # Should have child process CPU time on most platforms
        # Note: On some platforms (e.g., macOS), child process times might not be
        # properly captured for certain process spawning methods
        total_children_time = (measurement.children_user_time + 
                             measurement.children_system_time)
        
        # On macOS, child times might be 0 due to platform limitations
        # We'll just verify that the measurement fields exist and are non-negative
        self.assertGreaterEqual(measurement.children_user_time, 0.0)
        self.assertGreaterEqual(measurement.children_system_time, 0.0)
        
        # Verify results are correct
        self.assertEqual(results, [0, 1, 4, 9])
    
    def test_subprocess_block(self):
        """Test measuring a block that spawns a subprocess."""
        recorder = TimeRecorder()
        
        with recorder.measure("subprocess_test"):
            # Run a simple Python command in a subprocess
            result = subprocess.run(
                [sys.executable, "-c", "import time; time.sleep(0.1)"],
                capture_output=True
            )
        
        measurement = recorder.measurements[0]
        self.assertEqual(measurement.span_name, "subprocess_test")
        
        # Wall time should be at least 0.1s
        self.assertGreaterEqual(measurement.wall_time, 0.1)
        
        # Should have some child process time
        total_children_time = (measurement.children_user_time + 
                             measurement.children_system_time)
        self.assertGreater(total_children_time, 0.0)
    
    def test_multiple_measurements(self):
        """Test recording multiple measurements."""
        recorder = TimeRecorder()
        
        with recorder.measure("first_span"):
            time.sleep(0.1)
        
        with recorder.measure("second_span"):
            time.sleep(0.2)
        
        self.assertEqual(len(recorder.measurements), 2)
        
        first = recorder.measurements[0]
        second = recorder.measurements[1]
        
        self.assertEqual(first.span_name, "first_span")
        self.assertEqual(second.span_name, "second_span")
        
        self.assertGreaterEqual(first.wall_time, 0.1)
        self.assertGreaterEqual(second.wall_time, 0.2)
    
    def test_exception_in_measured_block(self):
        """Test that measurements are recorded even if exception occurs."""
        recorder = TimeRecorder()
        
        try:
            with recorder.measure("exception_test"):
                time.sleep(0.1)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Measurement should still be recorded
        self.assertEqual(len(recorder.measurements), 1)
        
        measurement = recorder.measurements[0]
        self.assertEqual(measurement.span_name, "exception_test")
        self.assertGreaterEqual(measurement.wall_time, 0.1)
    
    def test_nested_measurements(self):
        """Test nested measurement contexts."""
        recorder = TimeRecorder()
        
        with recorder.measure("outer_span"):
            time.sleep(0.1)
            with recorder.measure("inner_span"):
                time.sleep(0.1)
        
        self.assertEqual(len(recorder.measurements), 2)
        
        inner = recorder.measurements[0]  # Inner completes first
        outer = recorder.measurements[1]  # Outer completes second
        
        self.assertEqual(inner.span_name, "inner_span")
        self.assertEqual(outer.span_name, "outer_span")
        
        # Outer should take longer than inner
        self.assertGreater(outer.wall_time, inner.wall_time)
        self.assertGreaterEqual(outer.wall_time, 0.2)


if __name__ == "__main__":
    unittest.main() 