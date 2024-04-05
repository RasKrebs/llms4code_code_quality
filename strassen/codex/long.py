import numpy as np

def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return (n & (n-1) == 0) and n != 0

def strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Implement the Strassen algorithm for matrix multiplication.
    """
    # Check if the inputs are valid matrices
    if A.shape != B.shape or len(A.shape) != 2 or A.shape[0] != A.shape[1] or not is_power_of_two(A.shape[0]):
        raise ValueError("Invalid input matrices. They must be square, have equal dimensions, and the dimensions must be powers of 2.")

    # Base case: 2x2 matrix
    if A.shape[0] == 2:
        return np.dot(A, B)

    # Divide matrices into quarters
    size = A.shape[0] // 2
    A11, A12, A21, A22 = A[:size, :size], A[:size, size:], A[size:, :size], A[size:, size:]
    B11, B12, B21, B22 = B[:size, :size], B[:size, size:], B[size:, :size], B[size:, size:]

    # Compute the seven products
    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, B12 - B22)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)

    # Combine the products to get the final result
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    # Combine the quarters to get the final result
    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))