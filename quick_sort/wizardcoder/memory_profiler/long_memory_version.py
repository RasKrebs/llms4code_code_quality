
# IMPORTED FROM EVALUATE SCRIPT
from memory_profiler import profile
import os
import psutil

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

# BELOW IMPORTS ARE FROM THE SOLUTION
        
from typing import List, Union

@profile
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
