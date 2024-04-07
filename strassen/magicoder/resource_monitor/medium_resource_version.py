import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



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
        a = np.array(a)  # convert a into a numpy array
        b = np.array(b)  # convert b into a numpy array

        if len(a) == 1 and len(b) == 1:   # base case for small matrices where standard multiplication is used
            return a*b

        mid = len(a) // 2

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
def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim)).tolist() 
    B = np.random.randint(1, 10, size=(dim, dim)).tolist()
    
    # Perform matrix multiplication using the Strassen algorithm
    C = MatrixMultiplier(A, B).strassen(A, B)
    
    return C



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute()


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

