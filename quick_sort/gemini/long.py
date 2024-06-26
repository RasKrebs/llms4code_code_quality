def quick_sort(data: list[int]) -> list[int]:
    """Sorts a list of integers in ascending order using the quick sort algorithm.

    Args:
        data: The list of integers to be sorted.

    Returns:
        A new list containing the sorted integers.

    Raises:
        TypeError: If the input data contains non-integer elements.
    """

    if not data:
        return []

    try:
        pivot = data[0]
        left = [i for i in data[1:] if i <= pivot]
        right = [i for i in data[1:] if i > pivot]
    except TypeError as e:
        raise TypeError("Input list must contain only integers.") from e

    return quick_sort(left) + [pivot] + quick_sort(right)
