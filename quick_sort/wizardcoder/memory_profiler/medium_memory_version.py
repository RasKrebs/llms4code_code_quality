
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
def quicksort(lst):
    """
    This function takes last element as pivot,
    places the pivot element at its correct position in sorted array,
    and places all smaller (smaller than pivot) to left of pivot
    and all greater elements to right of pivot.

    Input: Unsorted list of integers
    Output: Sorted list of integers
    """

    if len(lst) <= 1:
        return lst
    else:
        pivot = lst.pop() # Choose the last element as pivot

    items_greater = []
    items_lower = []

    for item in lst:
        if item > pivot:
            items_greater.append(item)  # Elements greater than the pivot are placed in 'items_greater' list
        else:
            items_lower.append(item)   # Elements lower than or equal to the pivot are placed in 'items_lower' list

    # The sorted list is obtained by recursively sorting 'items_lower' and 'items_greater', and concatenating them with the pivot in between.
    return quicksort(items_lower) + [pivot] + quicksort(items_greater)
