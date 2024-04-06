import numpy as np

def split(matrix):
    """Split a given matrix into quarters."""
    row, col = matrix.shape
    mid_row, mid_col = row//2, col//2

    return matrix[:mid_row, :mid_col], matrix[:mid_row, mid_col:], \
           matrix[mid_row:, :mid_col], matrix[mid_row:, mid_col:]

def strassen(A, B):
    """Multiply two matrices using Strassen's algorithm."""
    # Base case when size of the matrices is 1x1
    if len(A) == 1:
        return A * B

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(A)
    e, f, g, h = split(B)

    # Computing the 7 products: p1 to p7
    p1 = strassen(a, (f - h))
    p2 = strassen((a + b), h)
    p3 = strassen((c + d), e)
    p4 = strassen(d, (g - e))
    p5 = strassen((a + d), (e + h))
    p6 = strassen((b - d), (g + h))
    p7 = strassen((a - c), (e + f))

    # Computing the values of the 4 quadrants of the final matrix c
    c1 = p5 + p4 - p2 + p6
    c2 = p1 + p2
    c3 = p3 + p4
    c4 = p1 + p5 - p3 - p7

    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    top = np.hstack((c1, c2))
    bottom = np.hstack((c3, c4))
    return np.vstack((top, bottom))
