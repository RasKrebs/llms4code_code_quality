def quicksort(arr):
    """
    This function takes a list of integers as input and returns a sorted version of the list using the QuickSort algorithm.

    Parameters:
    arr (list): The list of integers to be sorted.

    Returns:
    list: The sorted list of integers.
    """
    if len(arr) <= 1:
        # Base case: if the list is of length 1 or empty, it's already sorted
        return arr
    else:
        pivot = arr[len(arr) // 2]
        # Divide the list into three parts: less than pivot, equal to pivot, and greater than pivot
        less = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        # Recursively sort the 'less' and 'greater' lists and combine the results
        return quicksort(less) + middle + quicksort(greater)