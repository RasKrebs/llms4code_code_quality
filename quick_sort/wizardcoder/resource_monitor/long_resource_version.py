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



from typing import List, Union

def quick_sort(input_list: List[Union[int, float]]) -> List[Union[int, float]]:
    """Implementation of the Quick sort Algorithm.

    This function implements the quick sort algorithm to sort a list
    of integers or floats in ascending order. It includes error checking for invalid inputs,
    and uses type hinting for better readability and maintainability.

    Parameters:
        input_list (List[Union[int, float]]): List of numbers to be sorted. Can be int or float.

    Returns:
        List[Union[int, float]]: The sorted list in ascending order.

    Raises:
        TypeError: If the input is not a list or if the elements of the list are not numbers (integers or floats).
    """

    # Check for valid input type
    if not isinstance(input_list, list):
        raise TypeError("Input must be a list.")

    # Check that all elements in the list are numbers
    for i in input_list:
        if not isinstance(i, (int, float)):
            raise ValueError("All elements in the list must be numbers (integers or floats).")

    n = len(input_list)

    # If the list contains one or no element, it's already sorted
    if n <= 1:
        return input_list
    else:
        pivot = input_list.pop()

        items_greater = []
        items_lower = []

        for item in input_list:
            if item > pivot:
                items_greater.append(item)

            else:
                items_lower.append(item)

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

