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


from typing import List, Any


def check_valid_matrices(A: Any, B: Any) -> bool:
    """
    Check if the input matrices A and B are valid for matrix multiplication.
    
    Args:
    A: First matrix to be multiplied.
    B: Second matrix to be multiplied.
    
    Returns:
    bool: True if matrices are valid, False otherwise.
    """
    if isinstance(A, list) and isinstance(B, list):
        if len(A) == len(B) and len(A[0]) == len(B[0]):
            if len(A) == len(A[0]) and len(B) == len(B[0]):
                n = len(A)
                return n & (n - 1) == 0  # Check if dimensions are power of 2
    return False


def strassen_matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    Strassen algorithm for matrix multiplication.
    
    Args:
    A: First matrix to be multiplied.
    B: Second matrix to be multiplied.
    
    Returns:
    List[List[int]]: Product of the two matrices A and B.
    """
    n = len(A)
    
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    new_size = n // 2
    
    # Divide matrices into submatrices
    A11 = [row[:new_size] for row in A[:new_size]]
    A12 = [row[new_size:] for row in A[:new_size]]
    A21 = [row[:new_size] for row in A[new_size:]]
    A22 = [row[new_size:] for row in A[new_size:]]
    
    B11 = [row[:new_size] for row in B[:new_size]]
    B12 = [row[new_size:] for row in B[:new_size]]
    B21 = [row[:new_size] for row in B[new_size:]]
    B22 = [row[new_size:] for row in B[new_size:]]
    
    # Calculate the 7 products using Strassen's method
    M1 = strassen_matrix_multiply(add_matrices(A11, A22), add_matrices(B11, B22))
    M2 = strassen_matrix_multiply(add_matrices(A21, A22), B11)
    M3 = strassen_matrix_multiply(A11, subtract_matrices(B12, B22))
    M4 = strassen_matrix_multiply(A22, subtract_matrices(B21, B11))
    M5 = strassen_matrix_multiply(add_matrices(A11, A12), B22)
    M6 = strassen_matrix_multiply(subtract_matrices(A21, A11), add_matrices(B11, B12))
    M7 = strassen_matrix_multiply(subtract_matrices(A12, A22), add_matrices(B21, B22))
    
    # Calculate resulting submatrices   
    C11 = add_matrices(subtract_matrices(add_matrices(M1, M4), M5), M7)
    C12 = add_matrices(M3, M5)
    C21 = add_matrices(M2, M4)
    C22 = add_matrices(add_matrices(subtract_matrices(M1, M2), M3), M6)
    
    # Combine submatrices into result matrix
    result_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(new_size):
        for j in range(new_size):
            result_matrix[i][j] = C11[i][j]
            result_matrix[i][j + new_size] = C12[i][j]
            result_matrix[i + new_size][j] = C21[i][j]
            result_matrix[i + new_size][j + new_size] = C22[i][j]
    
    return result_matrix


def add_matrices(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    Add two matrices element-wise.
    
    Args:
    A: First matrix.
    B: Second matrix.
    
    Returns:
    List[List[int]]: Sum of the two matrices.
    """
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[i][j] + B[i][j]
    return result


def subtract_matrices(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    Subtract two matrices element-wise.
    
    Args:
    A: First matrix.
    B: Second matrix.
    
    Returns:
    List[List[int]]: Difference of the two matrices.
    """
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[i][j] - B[i][j]
    return result

    
    
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

