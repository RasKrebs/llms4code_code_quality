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
    Sorts an array of integers in ascending order using the Quick Sort algorithm.

    Parameters:
    arr (list of int): The list of integers to be sorted.

    Returns:
    list of int: The sorted list of integers.

    """
    # Base case: if list is empty or contains a single element, it's already 'sorted'
    if len(arr) <= 1:
        return arr
    else:
        # Choosing the last element in the list as the pivot
        pivot = arr.pop()
      
        # Two sub-arrays to hold elements less than and greater than the pivot
        items_lower = []
        items_greater = []
      
        # Partitioning the array based on the pivot
        for item in arr:
            if item <= pivot:
                items_lower.append(item)
            else:
                items_greater.append(item)
      
        # The recursive step: quick_sort() is called for the two sub-arrays and the results are combined with the pivot into a new list
        return quick_sort(items_lower) + [pivot] + quick_sort(items_greater)
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

