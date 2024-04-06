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



def quick_sort(arr):
    """
    This function takes a list of integers as input and returns a sorted version of the list using the QuickSort algorithm.

    Parameters:
    arr (list): A list of integers.

    Returns:
    list: A sorted version of the input list.
    """

    # If the list has less than 2 elements, it is already sorted
    if len(arr) < 2:
        return arr

    # Select the pivot element (the first element in this case)
    pivot = arr[0]

    # Partition the list into two sub-lists: one with elements less than the pivot, and one with elements greater than the pivot
    less = [i for i in arr[1:] if i <= pivot]
    greater = [i for i in arr[1:] if i > pivot]

    # Recursively sort the sub-lists and concatenate the results
    return quick_sort(less) + [pivot] + quick_sort(greater)

# Test the function
print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
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

