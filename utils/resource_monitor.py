import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}

# --- DATA HERE ---

# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)


# --- MAIN CODE ---



# --- EXECUTE HERE ---


if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process
    output = execute(x)

    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))
