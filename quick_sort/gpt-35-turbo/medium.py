def quick_sort(arr):
    """
    Sorts a list of integers using the quick sort algorithm.

    Args:
    arr (list): List of integers to be sorted

    Returns:
    list: Sorted list of integers
    """

    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]

        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)
