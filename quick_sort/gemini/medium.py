"""
This function implements the Quick Sort algorithm to sort a list of integers in ascending order.

Args:
  data: The list of integers to be sorted.

Returns:
  A new list containing the sorted integers.
"""


def quick_sort(data):
  """
  Sorts a list of integers in ascending order using the Quick Sort algorithm.

  The Quick Sort algorithm works by recursively partitioning the list into two sub-lists:
    - A sub-list containing elements less than or equal to a pivot element.
    - A sub-list containing elements greater than the pivot element.

  The pivot element is typically chosen as the last element of the list. The sub-lists are then
  sorted recursively and concatenated to form the final sorted list.

  Args:
    data: The list of integers to be sorted.

  Returns:
    A new list containing the sorted integers.
  """

  if len(data) <= 1:
    return data

  pivot = data[-1]  # Choose the last element as the pivot
  left = [i for i in data[:-1] if i <= pivot]  # Left sub-list with elements <= pivot
  right = [i for i in data[:-1] if i > pivot]  # Right sub-list with elements > pivot

  return quick_sort(left) + quick_sort(right)  # Recursively sort sub-lists and concatenate
