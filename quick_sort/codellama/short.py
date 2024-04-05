def partition(arr, low, high):
    # Select the last element as pivot
    pivot = arr[high]

    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1

            # Swap elements at indices i and j
            arr[i], arr[j] = arr[j], arr[i]

    # Swap the pivot element with the element at index (i + 1)
    arr[i+1], arr[high] = arr[high], arr[i+1]

    return i+1

def quickSort(arr, low, high):
    if low < high:
        # Partition the array and get the pivot index
        pi = partition(arr, low, high)

        # Recursively sort elements before and after the pivot
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)