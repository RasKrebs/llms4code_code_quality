import numpy as np
from math import floor

def strassen_matrix_mult(A, B):
    def splitMatrix(mat):
        row, col = mat.shape
        return mat[:row//2, :col//2], mat[:row//2, col//2:], mat[row//2:, :col//2], mat[row//2:, col//2:]

    if len(A) == 1:
        return A * B

    a, b, c, d = splitMatrix(A)
    e, f, g, h = splitMatrix(B)

    p1 = strassen_matrix_mult(a, f - h)
    p2 = strassen_matrix_mult(a + b, h)
    p3 = strassen_matrix_mult(c + d, e)
    p4 = strassen_matrix_mult(d, g - e)
    p5 = strassen_matrix_mult(a + d, e + h)
    p6 = strassen_matrix_mult(b - d, g + h)
    p7 = strassen_matrix_mult(a - c, e + f)

    res11 = p5 + p4 - p2 + p6
    res12 = p1 + p2
    res21 = p3 + p4
    res22 = p1 + p5 - p3 - p7

    return np.vstack((np.hstack((res11, res12)), np.hstack((res21, res22))))
