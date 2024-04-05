from typing import List
import numpy as np

def strassen_matrix_mult(matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
    # Get the size of the square matrices
    size = len(matrix1)

    # Base case for small matrices
    if size == 1:
        return [[matrix1[0][0] * matrix2[0][0]]]

    # Divide each matrix into four sub-matrices
    mid_size = size // 2
    a, b, c, d = split_matrix(matrix1)
    e, f, g, h = split_matrix(matrix2)

    # Recursive calls for the seven products
    p1 = strassen_matrix_mult(a, subtract_matrices(f, h))
    p2 = strassen_matrix_mult(add_matrices(a, b), h)
    p3 = strassen_matrix_mult(add_matrices(c, d), e)
    p4 = strassen_matrix_mult(d, subtract_matrices(g, e))
    p5 = strassen_matrix_mult(add_matrices(a, d), add_matrices(e, h))
    p6 = strassen_matrix_mult(subtract_matrices(b, d), add_matrices(g, h))
    p7 = strassen_matrix_mult(subtract_matrices(a, c), add_matrices(e, f))

    # Compute the four sums (or differences) required for the next step
    c1 = add_matrices(subtract_matrices(add_matrices(p5, p4), p2), p6)
    c2 = add_matrices(p1, p2)
    c3 = add_matrices(p3, p4)
    c4 = subtract_matrices(subtract_matrices(add_matrices(p5, p1), p3), p7)

    # Combine the four products into a single resulting matrix
    top_left = c1[:mid_size] + c2[:mid_size]
    top_right = c1[mid_size:] + c2[mid_size:]
    bottom_left = c3[:mid_size] + c4[:mid_size]
    bottom_right = c3[mid_size:] + c4[mid_size:]

    # Return the resulting matrix
    return top_left + bottom_left, top_right + bottom_right


def split_matrix(matrix: List[List[int]]) -> List[List[List[int]]]:
    """Split a square matrix into four equal parts."""

    mid = len(matrix) // 2
    return [row[:mid] for row in matrix[:mid]], \
           [row[mid:] for row in matrix[:mid]], \
           [row[:mid] for row in matrix[mid:]], \
           [row[mid:] for row in matrix[mid:]]

def add_matrices(matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
    """Add two matrices together."""

    return [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1))] for i in range(len(matrix1))]

def subtract_matrices(matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
    """Subtract the second matrix from the first matrix."""

    return [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1))] for i in range(len(matrix1))]