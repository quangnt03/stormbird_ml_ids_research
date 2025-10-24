"""
Resource monitoring utilities for tracking CPU, memory, and GPU usage during training and inference.
"""

import time
import psutil
import traceback
from functools import wraps


def get_gpu_info():
    """
    Get GPU utilization and memory usage.
    Returns tuple of (gpu_util_percent, gpu_memory_used_mb) or (None, None) if GPU not available.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_used_mb = mem_info.used / 1024 / 1024

        pynvml.nvmlShutdown()
        return gpu_util, gpu_mem_used_mb
    except:
        # GPU not available or pynvml not installed
        return None, None


def measure_resources(func, *args, **kwargs):
    """
    Measure resource usage during function execution.

    Returns:
        dict: {
            'result': function return value,
            'elapsed_time_sec': execution time in seconds,
            'cpu_usage_percent': CPU usage percentage,
            'peak_memory_mb': peak memory usage in MB,
            'gpu_util_percent': GPU utilization percentage (if available),
            'gpu_mem_mb': GPU memory used in MB (if available)
        }
    """
    # Get process
    process = psutil.Process()

    # Initial measurements
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Start monitoring
    start_time = time.time()
    cpu_percent_start = process.cpu_percent(interval=None)

    # Execute function
    result = func(*args, **kwargs)

    # End monitoring
    elapsed_time = time.time() - start_time
    cpu_percent_end = process.cpu_percent(interval=None)
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Get average CPU usage
    cpu_usage = (cpu_percent_start + cpu_percent_end) / 2

    # Peak memory
    peak_memory = max(initial_memory, final_memory)

    # GPU info
    gpu_util, gpu_mem = get_gpu_info()

    return {
        'result': result,
        'elapsed_time_sec': elapsed_time,
        'cpu_usage_percent': cpu_usage,
        'peak_memory_mb': peak_memory,
        'gpu_util_percent': gpu_util,
        'gpu_mem_mb': gpu_mem
    }


class ResourceMonitor:
    """Context manager for monitoring resources during execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.cpu_samples = []
        self.memory_samples = []

    def __enter__(self):
        self.start_time = time.time()
        # Take initial sample
        self.cpu_samples.append(self.process.cpu_percent(interval=None))
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        # Take final sample
        self.cpu_samples.append(self.process.cpu_percent(interval=None))
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)

    def get_metrics(self):
        """Get collected metrics."""
        gpu_util, gpu_mem = get_gpu_info()

        return {
            'elapsed_time_sec': self.end_time - self.start_time if self.end_time else 0,
            'cpu_usage_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'peak_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'gpu_util_percent': gpu_util,
            'gpu_mem_mb': gpu_mem
        }
