def quick_sort(arr: list[int]) -> list[int]:
    """
    Sorts a list of integers in ascending order using the quick sort algorithm.
    
    Parameters:
    arr (list[int]): The list of integers to sort.

    Returns:
    list[int]: A new list containing the sorted integers.

    Raises:
    ValueError: If the input is not a list of integers.
    """
    # Validate input
    if not isinstance(arr, list) or not all(isinstance(x, int) for x in arr):
        raise ValueError("Input must be a list of integers.")

    _quick_sort_recursive(arr, 0, len(arr) - 1)
    return arr

def _quick_sort_recursive(arr: list[int], low: int, high: int) -> None:
    """
    A recursive helper function to implement quick sort algorithm.
    
    Parameters:
    arr (list[int]): The list of integers to sort.
    low (int): The starting index of the subset of the list to be sorted.
    high (int): The ending index of the subset of the list to be sorted.
  
    This function modifies the input list in place.
    """
    if low < high:
        # Partition the list
        pi = _partition(arr, low, high)

        # Recursively sort the partitions
        _quick_sort_recursive(arr, low, pi - 1)
        _quick_sort_recursive(arr, pi + 1, high)

def _partition(arr: list[int], low: int, high: int) -> int:
    """
    Helper function to partition the list around a pivot.

    Parameters:
    arr (list[int]): The list to partition.
    low (int): The starting index of the subset of the list to partition.
    high (int): The ending index of the subset of the list to partition.

    Returns:
    int: The index of the pivot after partitioning.
    """
    pivot = arr[high]
    i = low - 1  # Index of smaller element
  
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Swap

    arr[i+1], arr[high] = arr[high], arr[i+1]  # Swap pivot to correct position
    return (i+1)

