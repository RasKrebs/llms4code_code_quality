
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
def quick_sort(arr: List[int]) -> List[int]:
    """
    Sorts a list of integers using the quick sort algorithm.
    
    Args:
    arr (List[int]): List of integers to be sorted
    
    Returns:
    List[int]: Sorted list of integers
    
    Raises:
    ValueError: If input is not a list
    ValueError: If input list contains non-integer elements
    """

    # Check if input is a list
    if not isinstance(arr, list):
        raise ValueError("Input must be a list of integers")

    # Check if input list contains non-integer elements
    if not all(isinstance(x, int) for x in arr):
        raise ValueError("Input list must contain only integers")

    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)

# Test the function
try:
    arr = [3, 2, 1, 5, 4]
    sorted_arr = quick_sort(arr)
    print(sorted_arr)
except ValueError as e:
    print(e)
