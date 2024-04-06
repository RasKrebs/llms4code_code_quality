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

def check_matrices(A, B):
    if not isinstance(A, (list, np.ndarray)) or not isinstance(B, (list, np.ndarray)):
        raise TypeError("Inputs should be lists, numpy arrays, or pandas dataframes")

    if len(A) == 0 or len(B) == 0:
        raise ValueError("Input matrices should not be empty")

    if len(A) != len(A[0]) or len(B) != len(B[0]):
        raise ValueError("Matrices should be square matrices")

    if len(A) != len(B):
        raise ValueError("Matrices should have the same dimensions for multiplication")

    n = len(A)
    if (n & (n - 1)) != 0:
        raise ValueError("Matrix dimensions should be powers of 2")


def strassen_matrix_multiply(A, B):
    check_matrices(A, B)

    n = len(A)

    if n <= 2:
        return np.dot(A, B)

    # Divide matrices into submatrices
    k = n // 2
    A11 = A[:k, :k]
    A12 = A[:k, k:]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    B11 = B[:k, :k]
    B12 = B[:k, k:]
    B21 = B[k:, :k]
    B22 = B[k:, k:]

    # Calculate Strassen algorithm matrices
    M1 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    M2 = strassen_matrix_multiply(A21 + A22, B11)
    M3 = strassen_matrix_multiply(A11, B12 - B22)
    M4 = strassen_matrix_multiply(A22, B21 - B11)
    M5 = strassen_matrix_multiply(A11 + A12, B22)
    M6 = strassen_matrix_multiply(A21 - A11, B11 + B12)
    M7 = strassen_matrix_multiply(A12 - A22, B21 + B22)

    # Calculate result submatrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine submatrices to get result
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C
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

