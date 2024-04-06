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



def quicksort(arr):
    # Base case: if the list is of length 0 or 1, it's already sorted.
    if len(arr) <= 1:
        return arr
    else:
        # Select a pivot element. Here we choose the element in the middle of the list.
        # Other strategies could involve choosing the first or last element.
        pivot = arr[len(arr) // 2]
        
        # Partition the array into three lists: less than, equal to, and greater than the pivot.
        less_than = [x for x in arr if x < pivot]
        equal_to = [x for x in arr if x == pivot]
        greater_than = [x for x in arr if x > pivot]
        
        # Recursively apply quicksort to less_than and greater_than,
        # and concatenate the results with equal_to in between.
        return quicksort(less_than) + equal_to + quicksort(greater_than)
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

