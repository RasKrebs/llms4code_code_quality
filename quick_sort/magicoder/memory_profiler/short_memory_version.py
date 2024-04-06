
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
    if len(arr) <= 1:  # base case, arrays with 0 or 1 element are already sorted
        return arr
    else:
        pivot = arr[len(arr) // 2]   # pick the middle element as pivot point

        less_than_pivot = [x for x in arr if x < pivot]  # elements less than pivot
        equal_to_pivot = [x for x in arr if x == pivot] # elements equal to pivot
        greater_than_pivot = [x for x in arr if x > pivot]   # elements greater than pivot

        return quicksort(less_than_pivot) + equal_to_pivot + quicksort(greater_than_pivot)  # recursive call on each half
