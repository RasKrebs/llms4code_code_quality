
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
        
from typing import List

@profile
def quicksort(arr: List[int]) -> List[int]:
    """
    This function sorts a list of integers using the quicksort algorithm.

    Parameters:
    arr (List[int]): The list of integers to be sorted.

    Returns:
    List[int]: The sorted list of integers.
    """
    if not all(isinstance(i, int) for i in arr):
        raise ValueError("Input list must contain only integers.")

    if len(arr) <= 1:
        # Base case: a list of zero or one elements is already sorted
        return arr
    else:
        pivot = arr[0]
        # Sub-array of all the elements less than the pivot
        less = [i for i in arr[1:] if i <= pivot]
        # Sub-array of all the elements greater than the pivot
        greater = [i for i in arr[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)
