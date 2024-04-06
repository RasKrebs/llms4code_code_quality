
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
def quicksort(numbers):
    """This function takes a list of numbers and returns the sorted version using QuickSort algorithm."""

    # check if input is a list
    if not isinstance(numbers, list):
        raise ValueError("Input must be a list.")

    # check if all elements in list are integers
    for number in numbers:
        if not isinstance(number, int):
            raise TypeError('List can only contain integer values.')

    # base case
    if len(numbers) <= 1:
        return numbers

    pivot = numbers[0]
    left_side = [i for i in numbers[1:] if i < pivot]
    right_side = [i for i in numbers[1:] if i >= pivot]

    return quicksort(left_side) + [pivot] + quicksort(right_side)
