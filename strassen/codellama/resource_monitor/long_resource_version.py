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



def strassen_multiply(A, B):
    """
    Implement the Strassen algorithm for matrix multiplication in Python from scratch.
    Args:
        A (list of lists or numpy array or pandas dataframe): First input matrix.
        B (list of lists or numpy array or pandas dataframe): Second input matrix.
    Returns:
        list of lists: Product of the two matrices.
    """

    # Check if both matrices are square and have equal dimensions
    A_rows, A_cols = len(A), len(A[0])
    B_rows, B_cols = len(B), len(B[0])
    if A_rows != A_cols or B_rows != B_cols:
        raise ValueError("Input matrices must be square.")
    if A_rows != B_rows:
        raise ValueError("Input matrices must have equal dimensions.")

    # Check if matrix sizes are powers of 2
    n = A_rows
    if (n & (n - 1) == 0) and n != 0:
        return strassen_helper(A, B, n)
    else:
        raise ValueError("Matrix dimensions must be a power of 2.")

def strassen_helper(A, B, n):
    """
    Helper function to recursively multiply matrices using Strassen's algorithm.
    Args:
        A (list of lists or numpy array or pandas dataframe): First input matrix.
        B (list of lists or numpy array or pandas dataframe): Second input matrix.
        n (int): Dimension of the submatrices being multiplied.
    Returns:
        list of lists: Product of the two matrices.
    """
    # Base case for small matrices where standard multiplication is used
    if n <= 2:
        return [[sum(a * b for a, b in zip(A_row, B_col))
                 for B_col in zip(*B)] for A_row in A]

    # Divide the matrices into quadrants
    mid = n // 2
    A11 = list_of_lists_to_submatrix(A, 0, mid, 0, mid)
    A12 = list_of_lists_to_submatrix(A, 0, mid, mid, n)
    A21 = list_of_lists_to_submatrix(A, mid, n, 0, mid)
    A22 = list_of_lists_to_submatrix(A, mid, n, mid, n)

    B11 = list_of_lists_to_submatrix(B, 0, mid, 0, mid)
    B12 = list_of_lists_to_submatrix(B, 0, mid, mid, n)
    B21 = list_of_lists_to_submatrix(B, mid, n, 0, mid)
    B22 = list_of_lists_to_submatrix(B, mid, n, mid, n)

    # Calculate the 7 products required by Strassen's algorithm
    M1 = strassen_helper(add_matrices(A11, A22), add_matrices(B11, B22), mid)
    M2 = strassen_helper(add_matrices(A21, A22), B11, mid)
    M3 = strassen_helper(A11, subtract_matrices(B12, B22), mid)
    M4 = strassen_helper(A22, subtract_matrices(B21, B11), mid)
    M5 = strassen_helper(add_matrices(A11, A12), B22, mid)
    M6 = strassen_helper(subtract_matrices(A21, A11), add_matrices(B11, B12), mid)
    M7 = strassen_helper(subtract_matrices(A12, A22), add_matrices(B21, B22), mid)

    # Combine the subproducts to form the final product
    C11 = add_matrices(subtract_matrices(add_matrices(M1, M4), M5), M7)
    C12 = add_matrices(M3, M5)
    C21 = add_matrices(M2, M4)
    C22 = add_matrices(subtract_matrices(add_matrices(M1, M3), M2), M6)

    # Recombine the quadrants into a single matrix by stacking rows and columns
    return [row1 + row2 for row1, row2 in zip(C11, C12)] + [row1 + row2 for row1, row2 in zip(C21, C22)]

def list_of_lists_to_submatrix(input_list, start_row, end_row, start_col, end_col):
    """
    Extract a submatrix from a matrix represented as a list of lists.
    Args:
        input_list (list of lists or numpy array or pandas dataframe): Input matrix.
        start_row (int): Index of the first row to be extracted.
        end_row (int): Index of the last row to be extracted.
        start_col (int): Index of the first column to be extracted.
        end_col (int): Index of the last column to be extracted.
    Returns:
        list of lists: Extracted submatrix represented as a list of lists.
    """
    return [row[start_col:end_col] for row in input_list[start_row:end_row]]

def add_matrices(A, B):
    """
    Add two matrices element-wise (entry by entry).
    Args:
        A (list of lists or numpy array or pandas dataframe): First matrix.
        B (list of lists or numpy array or pandas dataframe): Second matrix.
    Returns:
        list of lists: Sum of the two matrices.
    """
    return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

def subtract_matrices(A, B):
    """
    Subtract one matrix from another element-wise (entry by entry).
    Args:
        A (list of lists or numpy array or pandas dataframe): First matrix.
        B (list of lists or numpy array or pandas dataframe): Second matrix.
    Returns:
        list of lists: Difference between the two matrices.
    """
    return [[a - b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

def main():
    # Example usage
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    result = strassen_multiply(A, B)
    print(result)
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

