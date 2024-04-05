import numpy as np

def split(matrix):
    """
    Splits a given matrix into quarters.
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def strassen(x, y):
    """
    Computes matrix product by divide and conquer approach, recursively.
    """
    if len(x) == 1:
        return x * y

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)

    # Computing the 7 products, recursively (p1, p2...p7).
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)        
    p3 = strassen(c + d, e)        
    p4 = strassen(d, g - e)        
    p5 = strassen(a + d, e + h)        
    p6 = strassen(b - d, g + h)  
    p7 = strassen(a - c, e + f) 

    # Re-combining the 7 products to get the final result.
    upper_left = p5 + p4 - p2 + p6
    upper_right = p1 + p2
    lower_left = p3 + p4
    lower_right = p1 + p5 - p3 - p7

    # Combining the 4 results into a single matrix by stacking horizontally and vertically.
    top = np.hstack((upper_left, upper_right))
    bottom = np.hstack((lower_left, lower_right))
    return np.vstack((top, bottom))