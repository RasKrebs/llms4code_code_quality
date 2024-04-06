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
    """Sorts a list of integers in ascending order using the quick sort algorithm.

    Args:
        arr (list): The list to be sorted.

    Returns:
        list: The sorted list.

    Raises:
        TypeError: If any element is not an integer.
        ValueError: If the input array is empty or contains a single element.
    """
    if len(arr) < 2:
        raise ValueError("Input array must contain at least two elements.")

    for x in arr:
        if type(x) != int:
            raise TypeError("All elements of the input array must be integers.")

    def partition(array, low, high):
        i = (low - 1)
        pivot = array[high]  # pivot

        for j in range(low, high):
            if array[j] <= pivot:
                i = i + 1
                array[i], array[j] = array[j], array[i]

        array[i + 1], array[high] = array[high], array[i + 1]
        return (i + 1)

    def quick_sort_helper(array, low, high):
        if low < high:
            partition_index = partition(array, low, high)

            quick_sort_helper(array, low, partition_index - 1)
            quick_sort_helper(array, partition_index + 1, high)

    try:
        quick_sort_helper(arr, 0, len(arr) - 1)
    except TypeError as e:
        return f"TypeError: {e}"
    except ValueError as e:
        return f"ValueError: {e}"

    return arr
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

