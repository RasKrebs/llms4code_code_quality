def quick_sort(arr):
    """Quick Sort Algorithm

    Takes a list of integers as input and returns a sorted version of the same list using the quick sort algorithm.

    Args:
        arr (list of int): The unsorted list.

    Returns:
        list of int: The sorted list.
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)