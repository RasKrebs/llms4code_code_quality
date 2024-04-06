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

def quick_sort(nums: List[int]) -> List[int]:
    """
    Sorts a list of integers in ascending order using the Quick Sort algorithm.

    Args:
    nums: A list of integers to be sorted.

    Returns:
    A new list containing the sorted integers.

    Raises:
    TypeError: If the input is not a list or if the list contains non-integer elements.
    """

    # Check if input is a list
    if not isinstance(nums, list):
        raise TypeError("Input should be a list of integers.")

    # Check if list contains only integers
    for num in nums:
        if not isinstance(num, int):
            raise TypeError("List should only contain integers.")

    # If the list is empty or contains only one element, it is already sorted
    if len(nums) <= 1:
        return nums

    # Choose the first element as the pivot
    pivot = nums[0]

    # Partition the list into two sublists
    less_than_pivot = [x for x in nums[1:] if x <= pivot]
    greater_than_pivot = [x for x in nums[1:] if x > pivot]

    # Recursively sort the sublists and combine the results
    return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)
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

