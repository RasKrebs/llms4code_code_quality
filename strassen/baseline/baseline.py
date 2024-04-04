"""This module contains the implementation of Strassen's algorithm for matrix multiplication.
The algorithm is implemented in the function `strassen` which takes two matrices as input and
returns the product of the two matrices. The implementation handles matrices of dimensions that
are a power of 2 and are square matrices.
"""

from typing import Union, List
import numpy as np
import pandas as pd

# Implementation of Strassen's algorithm for matrix multiplication
def validate_input(matrix_a: Union[List[List[int]], np.ndarray, pd.DataFrame],
                   matrix_b: Union[List[List[int]], np.ndarray, pd.DataFrame]):
    """Validate the input matrices for matrix multiplication.

    Args:
        matrix_a (Union[List[List[int]], np.ndarray]): Matrix A to be multiplied.
        matrix_b (Union[List[List[int]], np.ndarray]): Matrix B to be multiplied.
        
    returns:
        matrix_a (np.ndarray): Matrix A to be multiplied.
        matrix_b (np.ndarray): Matrix B to be multiplied.
    """
    # Validate data formats
    for matrix in [matrix_a, matrix_b]:
        if not isinstance(matrix, (list, np.ndarray, pd.DataFrame)):
            raise ValueError((f"Input matrix must be a list, numpy array or pandas DataFrame."
                              " Got {type(matrix)}"))

    # Transform the input matrices into numpy arrays
    if isinstance(matrix_a, list):
        matrix_a = np.array(matrix_a)
    elif isinstance(matrix_a, pd.DataFrame):
        matrix_a = matrix_a.to_numpy()

    if isinstance(matrix_b, list):
        matrix_b = np.array(matrix_b)
    elif isinstance(matrix_b, pd.DataFrame):
        matrix_b = matrix_b.to_numpy()

    # Validating dimensions
    if not validate_matrix_dimensions(matrix_a):
        raise ValueError("Matrix A dimensions must be a power of 2")

    if not validate_matrix_dimensions(matrix_b):
        raise ValueError("Matrix B dimensions must be a power of 2")

    return matrix_a, matrix_b

def is_power_of_two(dim:int):
    """Check if a number is a power of 2.

    Args:
        dim (int): Number in dimension to check.

    Returns:
        bool: True if the number is a power of 2, False otherwise.
    """
    return dim == 2**(dim - 1).bit_length()

def validate_matrix_dimensions(matrix: np.ndarray):
    """Validate if the dimensions of a matrix are a power of 2.

    Args:
        matrix (np.ndarray): Matrix to validate.

    Returns:
        bool: True if the dimensions are a power of 2, False otherwise.
    """
    # Get the shape of the matrix
    rows, cols = matrix.shape

    # Check if the matrix is a square matrix
    if rows != cols:
        raise ValueError("Matrix must be a square matrix")

    # Check if the dimensions are a power of 2
    return is_power_of_two(rows) and is_power_of_two(cols)

def split_matrix(matrix: np.ndarray):
    """Split the matrix into 4 quadrants.

    Args:
        matrix (np.ndarray): Input matrix to be split.

    Returns:
        Tuple[np.ndarray]: 4 quadrants of the input matrix.
    """
    # Get the shape of the matrix
    length = matrix.shape[0]

    # Split the matrix into 4 quadrants
    A11 = matrix[:length//2, :length//2]
    A12 = matrix[:length//2, length//2:]
    A21 = matrix[length//2:, :length//2]
    A22 = matrix[length//2:, length//2:]

    return A11, A12, A21, A22

def strassen(matrix_a: Union[List[List[int]], np.ndarray, pd.DataFrame],
            matrix_b: Union[List[List[int]], np.ndarray, pd.DataFrame]):
    """Strassen's algorithm for matrix multiplication.

    Args:
        matrix_a (Union[List[List[int]], np.ndarray]): Matrix A to be multiplied.
        matrix_b (Union[List[List[int]], np.ndarray]): Matrix B to be multiplied.
    """
    # Validate data formats
    matrix_a, matrix_b = validate_input(matrix_a, matrix_b)

    # if the length of the array is 1, then multiply the two matrices
    if len(matrix_a) == 1:
        return matrix_a * matrix_b

    # Split the first matrix into 4 parts
    a11, a12, a21, a22 = split_matrix(matrix_a)
    b11, b12, b21, b22 = split_matrix(matrix_b)

    # Calculate the 10 products
    S1 = b12 - b22
    S2 = a11 + a12
    S3 = a21 + a22
    S4 = b21 - b11
    S5 = a11 + a22
    S6 = b11 + b22
    S7 = a12 - a22
    S8 = b21 + b22
    S9 = a11 - a21
    S10 = b11 + b12

    # Calculate the 7 products
    P1 = strassen(a11,S1)
    P2 = strassen(S2,b22)
    P3 = strassen(S3,b11)
    P4 = strassen(a22,S4)
    P5 = strassen(S5,S6)
    P6 = strassen(S7,S8)
    P7 = strassen(S9,S10)

    # Combining products
    c11 = P5 +P4 -P2 +P6
    c12 = P1 +P2
    c21 = P3 +P4
    c22 = P5 +P1 -P3 -P7

    # Output matrix
    output = np.zeros((matrix_a.shape[1], matrix_b.shape[0]))

    # Assign the 4 quadrants to the output matrix
    output[:int(len(output)/2),:int(len(output)/2)] = c11
    output[:int(len(output)/2),int(len(output)/2):] = c12
    output[int(len(output)/2):,:int(len(output)/2)] = c21
    output[int(len(output)/2):,int(len(output)/2):] = c22

    return output
