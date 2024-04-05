import numpy as np

def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return (n & (n-1) == 0) and n != 0

def is_square(matrix: np.ndarray) -> bool:
    """Check if a matrix is square."""
    return matrix.shape[0] == matrix.shape[1]

def strassen(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Strassen algorithm for matrix multiplication.
    
    Parameters:
    matrix1 (np.ndarray): First matrix
    matrix2 (np.ndarray): Second matrix

    Returns:
    np.ndarray: Product of the two matrices
    """
    # Check if matrices are valid
    if not (isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray)):
        raise ValueError("Both matrices must be numpy arrays.")
    if not (is_square(matrix1) and is_square(matrix2)):
        raise ValueError("Both matrices must be square.")
    if not (is_power_of_two(matrix1.shape[0]) and is_power_of_two(matrix2.shape[0])):
        raise ValueError("Both matrices must have dimensions that are powers of 2.")
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have equal dimensions.")

    # Base case: If the matrix is 1x1, return the product of the elements
    if matrix1.shape[0] == 1:
        return matrix1 * matrix2

    # Split the matrices into quadrants
    size = matrix1.shape[0] // 2
    a, b, c, d = matrix1[:size, :size], matrix1[:size, size:], matrix1[size:, :size], matrix1[size:, size:]
    e, f, g, h = matrix2[:size, :size], matrix2[:size, size:], matrix2[size:, :size], matrix2[size:, size:]

    # Compute the 7 products, recursively
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    # Combine the products to get the final matrix
    result = np.empty_like(matrix1)
    result[:size, :size] = p5 + p4 - p2 + p6
    result[:size, size:] = p1 + p2
    result[size:, :size] = p3 + p4
    result[size:, size:] = p1 + p5 - p3 - p7

    return result