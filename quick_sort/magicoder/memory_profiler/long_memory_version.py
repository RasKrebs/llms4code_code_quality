
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
def quick_sort(lst):
    """Quick sort function that sorts a list in ascending order

    Args:
        lst (list[int]): A list of integers to be sorted.

    Raises:
        ValueError: If the input is not a list or contains non-integer values.

    Returns:
        List[int]: Sorted list in ascending order.

    """
    # Check if input is valid and convert it to a list if necessary
    try:
        lst = list(lst)

    except Exception as e:
        raise ValueError("Input must be iterable") from e

    for i in lst:
        if not isinstance(i, int):
            raise ValueError('All elements of the input sequence must be integers')

    # Perform Quick Sort algorithm
    @profile
    def _quick_sort(items):
        """Helper function to perform the actual quick sorting"""

        if len(items) <= 1:
            return items

        pivot = items[0]
        less = [x for x in items[1:] if x <= pivot]
        greater = [x for x in items[1:] if x > pivot]

        return _quick_sort(less) + [pivot] + _quick_sort(greater)

    return _quick_sort(lst)
