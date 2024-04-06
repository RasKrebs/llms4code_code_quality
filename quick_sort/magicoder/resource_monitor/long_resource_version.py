import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import random
import numpy as np
np.random.seed(42)


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



def quick_sort(lst):
    """Quick sort function that sorts a list in ascending order

    Args:
        lst (list[int]): A list of integers to be sorted.

    Raises:
        ValueError: If the input is not a list or contains non-integer values.

    Returns:
        List[int]: Sorted list in ascending order.

    """
    # Check if input is valid and convert it to a list if necessary
    try:
        lst = list(lst)

    except Exception as e:
        raise ValueError("Input must be iterable") from e

    for i in lst:
        if not isinstance(i, int):
            raise ValueError('All elements of the input sequence must be integers')

    # Perform Quick Sort algorithm
    def _quick_sort(items):
        """Helper function to perform the actual quick sorting"""

        if len(items) <= 1:
            return items

        pivot = items[0]
        less = [x for x in items[1:] if x <= pivot]
        greater = [x for x in items[1:] if x > pivot]

        return _quick_sort(less) + [pivot] + _quick_sort(greater)

    return _quick_sort(lst)
def execute():
    # Set the random seed for reproducibility
    random.seed(42)
    
    # Generate random data: a list of 10 integers
    arr = [random.randint(0, 100) for _ in range(10000)]
    
    # Attempt to sort the array using our quick_sort function
    sorted_arr = quick_sort(arr)



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute()


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

