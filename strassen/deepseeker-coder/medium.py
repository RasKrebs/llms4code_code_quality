import numpy as np
from typing import Union, List

def split_matrix(matrix: np.ndarray) -> tuple:
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def strassen(x: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    """
    Recursive function to calculate matrix product by Strassen's algorithm.
    Input: two nxn matrices x and y
    Output: nxn matrix, product of x and y
    """
    # Convert the input to numpy arrays for convenience
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    # Base case for 1x1 matrix
    if len(x) == 1:
        return x * y

    # Split matrices into quarters. This will be used in recursive calls
    a, b, c, d = split_matrix(x)
    e, f, g, h = split_matrix(y)

    # Computing the seven products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    return c