
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
        
@profile
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

    @profile
    def partition(array, low, high):
        i = (low - 1)
        pivot = array[high]  # pivot

        for j in range(low, high):
            if array[j] <= pivot:
                i = i + 1
                array[i], array[j] = array[j], array[i]

        array[i + 1], array[high] = array[high], array[i + 1]
        return (i + 1)

    @profile
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
