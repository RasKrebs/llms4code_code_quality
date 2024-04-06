
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
