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



from typing import List

def quicksort(arr: List[int]) -> List[int]:
    """
    This function sorts a list of integers using the quicksort algorithm.

    Parameters:
    arr (List[int]): The list of integers to be sorted.

    Returns:
    List[int]: The sorted list of integers.
    """
    if not all(isinstance(i, int) for i in arr):
        raise ValueError("Input list must contain only integers.")

    if len(arr) <= 1:
        # Base case: a list of zero or one elements is already sorted
        return arr
    else:
        pivot = arr[0]
        # Sub-array of all the elements less than the pivot
        less = [i for i in arr[1:] if i <= pivot]
        # Sub-array of all the elements greater than the pivot
        greater = [i for i in arr[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)
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

