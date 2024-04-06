import numpy as np
from typing import List

def strassen(a: List[List], b: List[List]) -> List[List]:
    # Checking if matrices are valid for multiplication
    rows_A = len(a)
    cols_A = len(a[0])
    rows_B = len(b)
    cols_B = len(b[0])

    if cols_A != rows_B:
        raise ValueError("Matrices are not valid for multiplication")

    # Converting lists to numpy arrays
    a, b = np.array(a), np.array(b)

    # Base case for recursion when we have 1x1 matrix
    if len(a) == 1:
        return [a[0] * b[0]]

    n = len(a)
    mid = n // 2

    # Partitioning the matrices into quadrants for Strassen multiplication
    a11, a12, a21, a22 = a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]
    b11, b12, b21, b2 = b[:mid, :mid], b[:mid, mid:], b[mid:, :mid], b[mid:, mid:]

    # Recursive calls for Strassen multiplication
    p1 = strassen(a11, sub_matrix(b12, b22))
    p2 = strassen(add_matrix(a11, a12), b22)
    p3 = strassen(add_matrix(a21, a22), b11)
    p4 = strassen(a22, sub_matrix(b21, b11))
    p5 = strassen(add_matrix(a11, a22), add_matrix(b11, b22))
    p6 = strassen(sub_matrix(a12, a22), add_matrix(b21, b22))
    p7 = strassen(sub_matrix(a11, a21), add_matrix(b11, b12))

    # Calculating the values for result matrix using Strassen formula
    c11 = add_matrix(add_matrix(p5, p4), sub_matrix(p6, p2))
    c12 = add_matrix(p1, p2)
    c21 = add_matrix(p3, p4)
    c22 = sub_matrix(add_matrix(p5, p1), add_matrix(p3, p7))

    # Merging the quadrants into single matrix
    result = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))).tolist()

    return result

# Helper function to add two matrices
def add_matrix(m1: List[List], m2: List[List]) -> List[List]:
    return (np.add(m1, m2)).tolist()

# Helper function to subtract two matrices
def sub_matrix(m1: List[List], m2: List[List]) -> List[List]:
    return (np.subtract(m1, m2)).tolist()
