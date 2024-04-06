
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
    """Implements Quick Sort Algorithm

    Parameters:
    arr (list[int]): The list to be sorted

    Returns:
    list[int]: A sorted version of the input list
    """

    # If the list has 0 or 1 items, it's already sorted
    if len(arr) <= 1:
        return arr

    # Choose pivot to be middle element of array
    pivot = arr[len(arr) // 2]

    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
