import numpy as np  # for numpy array manipulation
from typing import Union  # for type hinting

class MatrixMultiplier(object):
    def __init__(self, matrix1: Union[list,np.ndarray], matrix2: Union[list,np.ndarray]):
        """
        Initialize the matrix multiplier with two matrices.

        Args:
            matrix1 (Union[list, np.ndarray]): First input matrix to be multiplied.
            matrix2 (Union[list, np.ndarray]): Second input matrix to be multiplied.
        """
        self._matrix1 = np.asarray(matrix1)  # convert the inputs into numpy arrays for easy manipulation
        self._matrix2 = np.asarray(matrix2)

    def validate_inputs(self):
        """
        Validate if both matrices can be multiplied and are valid dimensions.
        """
        assert len(self._matrix1.shape) == 2, "Matrix 1 should have exactly two dimensions."
        assert len(self._matrix2.shape) == 2, "Matrix 2 should have exactly two dimensions."

        # check the number of columns in matrix 1 is equal to the number of rows in matrix 2
        assert self._matrix1.shape[1] == self._matrix2.shape[0], \
            f"Number of columns in Matrix 1 should be equal to Number of Rows in Matrix 2 but got {self._matrix1.shape[1]} and {self._matrix2.shape[0]} respectively."

        # check if both matrices are symmetric (not necessary for matrix multiplication)

    def strassen(self, a: np.ndarray, b: np.ndarray):
        """
        Implement Strassen's algorithm recursively.

        Args:
            a (np.ndarray): Input matrix 1.
            b (np.ndarray): Input matrix 2.

        Returns:
            result (np.ndarray): Result of the multiplication of two matrices.
        """
        if len(a) == 1 and len(b) == 1:   # base case for small matrices where standard multiplication is used
            return a*b

        mid = len(a[0]) // 2

        # split into quadrants
        a11, a12, a21, a22 = a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]
        b11, b12, b21, b22 = b[:mid, :mid], b[:mid, mid:], b[mid:, :mid], b[mid:, mid:]

        # perform calculations for each of the 7 products
        p1 = self.strassen(a11 + a22, b11 + b22)
        p2 = self.strassen((a21+a22), b11)
        p3 = self.strassen(a11, (b12-b22))
        p4 = self.strassen(a22, (b21-b11))
        p5 = self.strassen((a11+a12), b22)
        p6 = self.strassen((a21-a11),(b11+b12))
        p7 = self.strassen((a12-a22),(b21+b22))

        # compute the values of each quadrant in result matrix
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 - p2 + p3 + p6

        # combine the quadrants to form final result matrix
        return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    def multiply(self):
        """
        Multiply two matrices using Strassen's algorithm.

        Returns:
            product (np.ndarray): Result of the multiplication of two matrices.
        """
        self.validate_inputs()  # validate inputs before proceeding to multiplication

        result = self.strassen(self._matrix1, self._matrix2)

        return result
