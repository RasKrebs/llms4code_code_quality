def quick_sort(arr):
    """
    Sorts an array of integers in ascending order using the Quick Sort algorithm.

    Parameters:
    arr (list of int): The list of integers to be sorted.

    Returns:
    list of int: The sorted list of integers.

    """
    # Base case: if list is empty or contains a single element, it's already 'sorted'
    if len(arr) <= 1:
        return arr
    else:
        # Choosing the last element in the list as the pivot
        pivot = arr.pop()
      
        # Two sub-arrays to hold elements less than and greater than the pivot
        items_lower = []
        items_greater = []
      
        # Partitioning the array based on the pivot
        for item in arr:
            if item <= pivot:
                items_lower.append(item)
            else:
                items_greater.append(item)
      
        # The recursive step: quick_sort() is called for the two sub-arrays and the results are combined with the pivot into a new list
        return quick_sort(items_lower) + [pivot] + quick_sort(items_greater)
