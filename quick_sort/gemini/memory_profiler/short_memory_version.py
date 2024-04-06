
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
def quicksort(array):
    """Sorts an array using the quick sort algorithm."""

    if len(array) <= 1:
        return array  # Base case: arrays with 0 or 1 element are already sorted

    pivot_index = len(array) - 1  # Choose the last element as the pivot
    pivot = array[pivot_index]

    smaller, equal, larger = [], [], []
    for element in array:
        if element < pivot:
            smaller.append(element)
        elif element > pivot:
            larger.append(element)
        else:
            equal.append(element)

    return quicksort(smaller) + equal + quicksort(larger)
