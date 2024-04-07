import numpy as np

def strassen_matrix_mult(A, B):
    """
    Implements the Strassen algorithm for matrix multiplication in Python from scratch.

    Parameters:
    A (numpy array or list of lists) : Input matrix 1
    B (numpy array or list of lists) : Input matrix 2

    Returns:
    numpy array : The product of the two input matrices
    """

    # Convert inputs to numpy arrays if necessary
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Check that the matrices are square and have dimensions equal to powers of 2
    dim_A, dim_B = A.shape[0], B.shape[1]
    if not (np.isclose(dim_A, 2**int(np.log2(dim_A))) and np.isclose(dim_B, 2**int(np.log2(dim_B)))):
        raise ValueError("Matrices must have dimensions that are powers of 2")

    # Base case for small matrices (less than or equal to 1x1)
    if A.shape[0] == 1 and B.shape[1] == 1:
        return A * B

    # Split the input matrices into four quadrants
    mid_A = A.shape[0] // 2
    mid_B = B.shape[1] // 2

    A11, A12 = A[:mid_A, :mid_A], A[:mid_A, mid_A:]
    A21, A22 = A[mid_A:, :mid_A], A[mid_A:, mid_A:]

    B11, B12 = B[:mid_B, :mid_B], B[:mid_B, mid_B:]
    B21, B22 = B[mid_B:, :mid_B], B[mid_B:, mid_B:]

    # Perform the recursive multiplication steps using Strassen's formulae
    P1 = strassen_matrix_mult(A11 + A22, B11 + B22)
    P2 = strassen_matrix_mult(A21 + A22, B11)
    P3 = strassen_matrix_mult(A11, B12 - B22)
    P4 = strassen_matrix_mult(A22, B21 - B11)
    P5 = strassen_matrix_mult(A11 + A12, B22)
    P6 = strassen_matrix_mult(A21 - A11, B11 + B12)
    P7 = strassen_matrix_mult(A12 - A22, B21 + B22)

    # Combine the results into the final product matrix
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 + P3 - P2 + P6

    # Combine the four quadrants into a single matrix and return
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C
