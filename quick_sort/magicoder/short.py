def quicksort(arr):
    if len(arr) <= 1:  # base case, arrays with 0 or 1 element are already sorted
        return arr
    else:
        pivot = arr[len(arr) // 2]   # pick the middle element as pivot point

        less_than_pivot = [x for x in arr if x < pivot]  # elements less than pivot
        equal_to_pivot = [x for x in arr if x == pivot] # elements equal to pivot
        greater_than_pivot = [x for x in arr if x > pivot]   # elements greater than pivot

        return quicksort(less_than_pivot) + equal_to_pivot + quicksort(greater_than_pivot)  # recursive call on each half