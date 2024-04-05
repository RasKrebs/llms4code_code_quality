from typing import List

def quick_sort(nums: List[int]) -> List[int]:
    """
    Sorts a list of integers in ascending order using the Quick Sort algorithm.

    Args:
    nums: A list of integers to be sorted.

    Returns:
    A new list containing the sorted integers.

    Raises:
    TypeError: If the input is not a list or if the list contains non-integer elements.
    """

    # Check if input is a list
    if not isinstance(nums, list):
        raise TypeError("Input should be a list of integers.")

    # Check if list contains only integers
    for num in nums:
        if not isinstance(num, int):
            raise TypeError("List should only contain integers.")

    # If the list is empty or contains only one element, it is already sorted
    if len(nums) <= 1:
        return nums

    # Choose the first element as the pivot
    pivot = nums[0]

    # Partition the list into two sublists
    less_than_pivot = [x for x in nums[1:] if x <= pivot]
    greater_than_pivot = [x for x in nums[1:] if x > pivot]

    # Recursively sort the sublists and combine the results
    return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)