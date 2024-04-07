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
from typing import Union

def strassen_matrix_multiply(a: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> Union[np.ndarray, list]:
    """
    Function to implement the Strassen Matrix Multiplication Algorithm. It takes two matrices as input and returns their product.

    Args:
        a (Union[np.ndarray, list]): First matrix
        b (Union[np.ndarray, list]): Second matrix

    Returns:
        Union[np.ndarray, list]: Product of the two matrices
    """

    # Check if both input matrices are valid and can be multiplied
    if not isinstance(a, (np.ndarray, list)) or not isinstance(b, (np.ndarray, list)):
        raise TypeError("Both inputs must be either a numpy array or a list of lists")

    # Convert input matrices to numpy arrays for easier manipulation
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)

    # Check if both matrices are square and have equal dimensions
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise ValueError("Both input matrices must be square and have equal dimensions")

    # Check if both matrices can be divided into 4 sub-matrices (i.e., their dimensions must be powers of 2)
    if a.shape[0] & (a.shape[0] - 1):
        raise ValueError("Both input matrices must have dimensions that are powers of 2")

    # Base case: When the size of the matrices is 1x1, return their product
    if a.size == 4:
        return np.array([[a[0][0] * b[0][0]]])

    else:
        # Split the input matrices into four sub-matrices (quadrants)
        a11, a12, a21, a22 = split_matrix(a)
        b11, b12, b21, b22 = split_matrix(b)

        # Recursively calculate the 7 products (P1 to P7), which are needed for determining the product of A and B
        p1 = strassen_matrix_multiply(a11 + a22, b11 + b22)
        p2 = strassen_matrix_multiply(a21 + a22, b11)
        p3 = strassen_matrix_multiply(a11, b12 - b22)
        p4 = strassen_matrix_multiply(a22, b21 - b11)
        p5 = strassen_matrix_multiply(a11 + a12, b22)
        p6 = strassen_matrix_multiply(a21 - a11, b11 + b12)
        p7 = strassen_matrix_multiply(a12 - a22, b21 + b22)


        # Calculate the quadrants of the result matrix (c11 to c22), using the previously calculated products
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 + p3 - p2 + p6


        # Combine the quadrants to form the final result matrix and return it
        result_matrix = combine_matrices(c11, c12, c21, c22)

    return result_matrix


def split_matrix(m: np.ndarray):
    """
    Function to divide a given square matrix into four equal parts (quadrants).

    Args:
        m (np.ndarray): Square input matrix to be divided into quadrants


    Returns:
        tuple: Four numpy arrays, each representing one quadrant of the input matrix
    """

    mid_index = len(m) // 2

    return m[:mid_index, :mid_index], m[:mid_index, mid_index:], m[mid_index:, :mid_index], m[mid_index:, mid_index:]


def combine_matrices(c11: np.ndarray, c12: np.ndarray, c21: np.ndarray, c22: np.ndarray):
    """
    Function to combine four given quadrants into a final result matrix.

    Args:
        c11 (np.ndarray): Top-left quadrant of the result matrix
        c12 (np.ndarray): Top-right quadrant of the result matrix
        c21 (np.ndarray): Bottom-left quadrant of the result matrix
        c22 (np.ndarray): Bottom-right quadrant of the result matrix


    Returns:
        np.ndarray: The final result matrix, constructed by combining the four input quadrants
    """

    top = np.concatenate((c11, c12), axis=1)
    bottom = np.concatenate((c21, c22), axis=1)

    return np.concatenate((top, bottom), axis=0)
def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim)) 
    B = np.random.randint(1, 10, size=(dim, dim))
    
    # Perform matrix multiplication using the Strassen algorithm
    C = strassen_matrix_multiply(A, B)



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

