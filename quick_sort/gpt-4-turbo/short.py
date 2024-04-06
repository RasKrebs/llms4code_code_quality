def quicksort(arr):
    # Base case: if the list is of length 0 or 1, it's already sorted.
    if len(arr) <= 1:
        return arr
    else:
        # Select a pivot element. Here we choose the element in the middle of the list.
        # Other strategies could involve choosing the first or last element.
        pivot = arr[len(arr) // 2]
        
        # Partition the array into three lists: less than, equal to, and greater than the pivot.
        less_than = [x for x in arr if x < pivot]
        equal_to = [x for x in arr if x == pivot]
        greater_than = [x for x in arr if x > pivot]
        
        # Recursively apply quicksort to less_than and greater_than,
        # and concatenate the results with equal_to in between.
        return quicksort(less_than) + equal_to + quicksort(greater_than)
