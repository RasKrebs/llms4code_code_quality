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
from typing import Union, List

def split(matrix):
    """
    Splits a given matrix into 4 submatrices.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices.
    """
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return (matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:])

def strassen(A: np.ndarray, B: np.ndarray) -> Union[np.ndarray, None]:
    """
    Strassen's Matrix Multiplication Algorithm.

    Parameters:
        A (np.ndarray): The first matrix to be multiplied.
        B (np.ndarray): The second matrix to be multiplied.

    Returns:
        C (np.ndarray or None): The product of the two matrices if they are compatible, otherwise None.
    """
    # Check if A and B can be multiplied
    if A.shape[1] != B.shape[0]:
        return None

    # Base case: 1x1 matrix
    if len(A) == 1:
        return A * B

    # Split A and B into blocks
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    # 7 products of sub-matrices
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Compute the values of the new matrix C
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine the sub-matrices into a single matrix
    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim)) 
    B = np.random.randint(1, 10, size=(dim, dim))
    
    # Perform matrix multiplication using the Strassen algorithm
    C = strassen(A, B)



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

