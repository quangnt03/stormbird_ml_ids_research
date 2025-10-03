import time
import psutil
import threading
import tracemalloc


def measure_resources(func, *args, **kwargs):
    """
    Measure CPU, memory, GPU, and elapsed time of a given function
    (training or inference).
    """
    sample_interval = 0.1
    process = psutil.Process()
    rss_samples = []
    stop_flag = threading.Event()

    def monitor():
        while not stop_flag.is_set():
            mem_info = process.memory_info().rss  # RSS in bytes
            rss_samples.append(mem_info)
            time.sleep(sample_interval)

    cpu_before = process.cpu_percent(interval=None)
    # Start tracemalloc for allocations
    tracemalloc.start()

    # Start monitoring thread
    t = threading.Thread(target=monitor)
    t.start()

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # Stop monitoring
    stop_flag.set()
    t.join()

    # Stop tracemalloc
    tracemalloc.stop()
    cpu_after = process.cpu_percent(interval=None)
    # Sum up all memory allocated (including freed)

    # Ensure at least one RSS sample
    if not rss_samples:
        rss_samples.append(process.memory_info().rss)

    total_time = end_time - start_time
    peak_rss = max(rss_samples)
    # --- GPU monitoring (if available) ---
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)  # MB
    except Exception:
        gpu_util, gpu_mem = 0, 0

    return {
        "result": result,
        "elapsed_time_sec": total_time,
        "cpu_usage": cpu_after - cpu_before,
        "python_peak_memory_mb": peak_rss / (1024**2),
        "gpu_util_percent": gpu_util,
        "gpu_memory_used_mb": gpu_mem,
    }
