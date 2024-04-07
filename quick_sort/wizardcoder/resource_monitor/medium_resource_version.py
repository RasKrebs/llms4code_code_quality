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



def quicksort(lst):
    """
    This function takes last element as pivot,
    places the pivot element at its correct position in sorted array,
    and places all smaller (smaller than pivot) to left of pivot
    and all greater elements to right of pivot.

    Input: Unsorted list of integers
    Output: Sorted list of integers
    """

    if len(lst) <= 1:
        return lst
    else:
        pivot = lst.pop() # Choose the last element as pivot

    items_greater = []
    items_lower = []

    for item in lst:
        if item > pivot:
            items_greater.append(item)  # Elements greater than the pivot are placed in 'items_greater' list
        else:
            items_lower.append(item)   # Elements lower than or equal to the pivot are placed in 'items_lower' list

    # The sorted list is obtained by recursively sorting 'items_lower' and 'items_greater', and concatenating them with the pivot in between.
    return quicksort(items_lower) + [pivot] + quicksort(items_greater)
def execute():
    # Set the random seed for reproducibility
    random.seed(42)
    
    # Generate random data: a list of 10 integers
    arr = [random.randint(0, 100) for _ in range(10000)]
    
    # Attempt to sort the array using our quick_sort function
    sorted_arr = quicksort(arr)



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

