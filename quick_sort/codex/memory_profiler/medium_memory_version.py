
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
def quicksort(arr):
    """
    This function takes a list of integers as input and returns a sorted version of the list using the QuickSort algorithm.

    Parameters:
    arr (list): The list of integers to be sorted.

    Returns:
    list: The sorted list of integers.
    """
    if len(arr) <= 1:
        # Base case: if the list is of length 1 or empty, it's already sorted
        return arr
    else:
        pivot = arr[len(arr) // 2]
        # Divide the list into three parts: less than pivot, equal to pivot, and greater than pivot
        less = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        # Recursively sort the 'less' and 'greater' lists and combine the results
        return quicksort(less) + middle + quicksort(greater)
