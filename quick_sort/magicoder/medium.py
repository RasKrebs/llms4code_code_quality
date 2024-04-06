def quick_sort(arr):
    """
    This function sorts an array using the quicksort algorithm.

    :param arr: list - The unsorted integer array to be sorted
    :return: list - The sorted integer array
    """
    if len(arr) <= 1:  # Base case: arrays with one or zero elements are already sorted
        return arr
    else:
        pivot = arr[0]  # Choose the first element as the pivot

        # Sub-array of all items less than the pivot
        less_than_pivot = [i for i in arr[1:] if i <= pivot]

        # Sub-array of all items greater than the pivot
        greater_than_pivot = [i for i in arr[1:] if i > pivot]

        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)  # Recursive call on the two sub-arrays, concatenating with pivot in between
