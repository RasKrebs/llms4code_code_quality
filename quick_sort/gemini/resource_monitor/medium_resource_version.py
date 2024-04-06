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



"""
This function implements the Quick Sort algorithm to sort a list of integers in ascending order.

Args:
  data: The list of integers to be sorted.

Returns:
  A new list containing the sorted integers.
"""


def quick_sort(data):
  """
  Sorts a list of integers in ascending order using the Quick Sort algorithm.

  The Quick Sort algorithm works by recursively partitioning the list into two sub-lists:
    - A sub-list containing elements less than or equal to a pivot element.
    - A sub-list containing elements greater than the pivot element.

  The pivot element is typically chosen as the last element of the list. The sub-lists are then
  sorted recursively and concatenated to form the final sorted list.

  Args:
    data: The list of integers to be sorted.

  Returns:
    A new list containing the sorted integers.
  """

  if len(data) <= 1:
    return data

  pivot = data[-1]  # Choose the last element as the pivot
  left = [i for i in data[:-1] if i <= pivot]  # Left sub-list with elements <= pivot
  right = [i for i in data[:-1] if i > pivot]  # Right sub-list with elements > pivot

  return quick_sort(left) + quick_sort(right)  # Recursively sort sub-lists and concatenate
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

