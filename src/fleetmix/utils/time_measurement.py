"""
Time Measurement Module

This module measures time metrics for code blocks, including wall-clock and CPU times
for the process and its children.

Usage:
    recorder = TimeRecorder()
    with recorder.measure("my_span"):
        # Code block here
    print(recorder.measurements)

TimeMeasurement contains:
    - span_name: str, identifier for the code block
    - wall_time: float, elapsed real time in seconds
    - process_user_time: float, user CPU time of the process
    - process_system_time: float, system CPU time of the process
    - children_user_time: float, user CPU time of child processes
    - children_system_time: float, system CPU time of child processes
"""

from dataclasses import dataclass
from contextlib import contextmanager
import time
import os
from typing import List


@dataclass
class TimeMeasurement:
    """Data class containing all time measurements for a code block.
    
    Attributes:
        span_name: Identifier for the measured code block
        wall_time: Elapsed real time in seconds
        process_user_time: User CPU time of the process in seconds
        process_system_time: System CPU time of the process in seconds
        children_user_time: User CPU time of child processes in seconds
        children_system_time: System CPU time of child processes in seconds
    """
    span_name: str
    wall_time: float
    process_user_time: float
    process_system_time: float
    children_user_time: float
    children_system_time: float


class TimeRecorder:
    """Records time measurements for code blocks using context managers.
    
    The recorder stores all measurements in a list that can be accessed
    after the measurements are complete.
    """
    
    def __init__(self) -> None:
        """Initialize a new TimeRecorder with an empty measurements list."""
        self.measurements: List[TimeMeasurement] = []
    
    @contextmanager
    def measure(self, span_name: str):
        """Context manager to measure execution time of a code block.
        
        Args:
            span_name: A string identifier for the code block being measured
            
        Yields:
            None
            
        Example:
            recorder = TimeRecorder()
            with recorder.measure("data_processing"):
                # Your code here
                process_data()
        """
        # Record start times
        start_wall = time.perf_counter()
        start_times = os.times()
        
        try:
            yield
        finally:
            # Record end times
            end_wall = time.perf_counter()
            end_times = os.times()
            
            # Calculate deltas and store measurement
            measurement = TimeMeasurement(
                span_name=span_name,
                wall_time=end_wall - start_wall,
                process_user_time=end_times[0] - start_times[0],
                process_system_time=end_times[1] - start_times[1],
                children_user_time=end_times[2] - start_times[2],
                children_system_time=end_times[3] - start_times[3]
            )
            
            self.measurements.append(measurement) 