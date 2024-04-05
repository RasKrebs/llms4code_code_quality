import numpy as np

def is_power_of_2(n: int) -> bool:
    """Check if a number is a power of 2."""
    return (n != 0) and (n & (n-1) == 0)

def validate_matrix(matrix) -> bool:
    """Validate that a matrix is a square matrix with dimensions that are powers of 2."""
    # Check if matrix is a list of lists, a numpy array, or a pandas DataFrame
    if isinstance(matrix, list):
        rows = len(matrix)
        if rows == 0 or not all(isinstance(row, list) and len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("Invalid matrix format. The matrix should be a list of lists with equal row lengths.")
    elif isinstance(matrix, np.ndarray):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Invalid matrix format. The matrix should be a 2D numpy array with equal dimensions.")
    else:
        raise TypeError("Unsupported matrix type. The matrix should be a list of lists or a numpy array.")

    # Check if the matrix is square and has dimensions that are powers of 2
    dim = matrix.shape[0] if isinstance(matrix, np.ndarray) else len(matrix)
    if dim != matrix.shape[1] or not is_power_of_2(dim):
        raise ValueError("Matrix dimensions must be equal and powers of 2.")

    return True

def standard_matrix_multiplication(A, B):
    """Perform standard matrix multiplication."""
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def add_matrices(A, B):
    """Add two matrices."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def subtract_matrices(A, B):
    """Subtract two matrices."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def split_matrix(matrix):
    """Split a matrix into quadrants."""
    n = len(matrix)
    mid = n // 2
    top_left = [row[:mid] for row in matrix[:mid]]
    top_right = [row[mid:] for row in matrix[:mid]]
    bottom_left = [row[:mid] for row in matrix[mid:]]
    bottom_right = [row[mid:] for row in matrix[mid:]]
    return top_left, top_right, bottom_left, bottom_right

def strassen_algorithm(A, B):
    """Recursive implementation of the Strassen algorithm for matrix multiplication."""
    assert validate_matrix(A) and validate_matrix(B), "Input matrices are not valid for Strassen's algorithm."
    
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    elif n <= 2:
        return standard_matrix_multiplication(A, B)

    a11, a12, a21, a22 = split_matrix(A)
    b11, b12, b21, b22 = split_matrix(B)

    # Compute the 7 Strassen's products
    p1 = strassen_algorithm(add_matrices(a11, a22), add_matrices(b11, b22))
    p2 = strassen_algorithm(add_matrices(a21, a22), b11)
    p3 = strassen_algorithm(a11, subtract_matrices(b12, b22))
    p4 = strassen_algorithm(a22, subtract_matrices(b21, b11))
    p5 = strassen_algorithm(add_matrices(a11, a12), b22)
    p6 = strassen_algorithm(subtract_matrices(a21, a11), add_matrices(b11, b12))
    p7 = strassen_algorithm(subtract_matrices(a12, a22), add_matrices(b21, b22))

    # Combine the intermediary products to get the final result
    c11 = add_matrices(subtract_matrices(add_matrices(p1, p4), p5), p7)
    c12 = add_matrices(p3, p5)
    c21 = add_matrices(p2, p4)
    c22 = add_matrices(subtract_matrices(add_matrices(p1, p3), p2), p6)

    # Construct the resulting matrix from the quadrants
    top = [c11[i] + c12[i] for i in range(len(c11))]
    bottom = [c21[i] + c22[i] for i in range(len(c21))]
    return top + bottom