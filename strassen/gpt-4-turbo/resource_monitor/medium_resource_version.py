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
import pandas as pd

def is_power_of_2(n: int) -> bool:
    """Check if a number is a power of 2."""
    return (n != 0) and (n & (n-1)) == 0

def next_power_of_2(x: int) -> int:
    """Find the next power of 2 greater than or equal to x."""
    return 1 if x == 0 else 2**(x - 1).bit_length()

def validate_and_transform_input(matrix):
    """Validate the input matrix and transform it into np.array."""
    if isinstance(matrix, (pd.DataFrame, list)):
        # Convert to numpy array for uniformity
        matrix = np.array(matrix)
    elif not isinstance(matrix, np.ndarray):
        raise ValueError(f"Unsupported format. Expecting np.ndarray, pd.DataFrame, or list, got {type(matrix)}")
    
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    
    return matrix

def pad_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensure the matrix is square and dimensions are powers of 2 by padding zeros."""
    rows, cols = matrix.shape
    max_dim = max(rows, cols)
    new_dim = next_power_of_2(max_dim)
    
    if rows < new_dim or cols < new_dim:
        # Pad the matrix with zeros to get to the next power of 2
        padded_matrix = np.zeros((new_dim, new_dim))
        padded_matrix[:rows, :cols] = matrix
        return padded_matrix
    return matrix

def standard_matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Perform standard matrix multiplication (for small matrices)."""
    return np.dot(A, B)

def strassen_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Implement the Strassen algorithm for matrix multiplication."""
    n = A.shape[0]
    
    if n == 1:
        return A * B
    if n <= 2:
        # Base case: Use standard multiplication for 2x2 matrices
        return standard_matrix_multiplication(A, B)
    
    # Divide matrices into quarters
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    
    # Compute the 7 products, recursively.
    P1 = strassen_multiply(A11 + A22, B11 + B22)
    P2 = strassen_multiply(A21 + A22, B11)
    P3 = strassen_multiply(A11, B12 - B22)
    P4 = strassen_multiply(A22, B21 - B11)
    P5 = strassen_multiply(A11 + A12, B22)
    P6 = strassen_multiply(A21 - A11, B11 + B12)
    P7 = strassen_multiply(A12 - A22, B21 + B22)

    # Combine the intermediate products
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    # Combine quarters into the result matrix
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

def strassen_matrix_multiplication(A, B):
    """Convenience wrapper to apply Strassen multiplication on two input matrices (with validation and padding)."""
    A = validate_and_transform_input(A)
    B = validate_and_transform_input(B)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second.")
    
    # Ensure the matrices are square and dimensions are powers of 2
    A_padded = pad_matrix(A)
    B_padded = pad_matrix(B)

    result_padded = strassen_multiply(A_padded, B_padded)
    
    # Remove padding from the result if any
    result = result_padded[:A.shape[0], :B.shape[1]]
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
    C = strassen_matrix_multiplication(A, B)



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

