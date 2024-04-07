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



import numpy as np
from typing import List, Union

class Matrix:
    def __init__(self, input_array):
        self.matrix = input_array

    # Method to check if a matrix is symmetric or not
    @staticmethod
    def _is_symmetric(input_array) -> bool:
        return np.allclose(input_array, input_array.T, equal_nan=True)

    # Check if the matrices are valid and can be multiplied together
    @staticmethod
    def _check_validity(matrix1, matrix2):
        if not isinstance(matrix1, np.ndarray) or not isinstance(matrix2, np.ndarray):
            raise TypeError("Matrices must be NumPy arrays.")
        if len(matrix1) != len(matrix2):
            raise ValueError("Matrices must have the same dimensions.")
        if not Matrix._is_symmetric(matrix1) or not Matrix._is_symmetric(matrix2):
            raise ValueError("Matrices must be symmetric.")
        # Check if matrix dimensions are powers of 2 (required by Strassen algorithm)
        if len(matrix1) & (len(matrix1) - 1) != 0 or len(matrix2) & (len(matrix2) - 1) != 0:
            raise ValueError("Matrices must have dimensions that are powers of 2.")

    # Recursively perform Strassen algorithm for matrix multiplication
    def strassen_multiply(self, other):
        Matrix._check_validity(self.matrix, other.matrix)
        return Matrix._strassen_algorithm(self.matrix, other.matrix)

    @staticmethod
    def _strassen_algorithm(A: Union[List[list], np.ndarray], B: Union[List[list], np.ndarray]) -> List[list]:
        """Recursively perform Strassen's algorithm for matrix multiplication"""
        # Base case for small matrices (e.g., 2x2): use standard matrix multiplication method
        if len(A) <= 2:
            return A @ B

        n = len(A) // 2

        # Split input matrices into quadrants
        A11, A12 = A[:n, :n], A[:n, n:]
        A21, A22 = A[n:, :n], A[n:, n:]
        B11, B12 = B[:n, :n], B[:n, n:]
        B21, B22 = B[n:, :n], B[n:, n:]

        # Calculate intermediate values
        M1 = Matrix._strassen_algorithm(A11 + A22, B11 + B22)
        M2 = Matrix._strassen_algorithm(A21 + A22, B11)
        M3 = Matrix._strassen_algorithm(A11, B12 - B22)
        M4 = Matrix._strassen_algorithm(A22, B21 - B11)
        M5 = Matrix._strassen_algorithm(A11 + A12, B22)
        M6 = Matrix._strassen_algorithm(A21 - A11, B11 + B12)
        M7 = Matrix._strassen_algorithm(A12 - A22, B21 + B22)

        # Compute intermediate quadrants of the result matrix
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        # Combine the quadrants into a full matrix using NumPy's concatenate and reshape operations
        result_matrix = np.concatenate((np.concatenate((C11, C12), axis=1),
                                        np.concatenate((C21, C22), axis=1)), axis=0)

        return result_matrix
def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim)) 
    B = np.random.randint(1, 10, size=(dim, dim))
    
    # Perform matrix multiplication using the Strassen algorithm
    mat = Matrix._strassen_algorithm(A,B)
    



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

