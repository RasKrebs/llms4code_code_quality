import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}

import numpy as np
import psutil
import os
import threading
import multiprocessing

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


def multiply_matrices(A, B):
    # Check if dimensions are compatible
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise Exception("Incompatible matrix sizes for multiplication")

    # Initialize resulting matrix
    C = [[0 for i in range(cols_B)] for j in range(rows_A)]

    # Perform standard matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


def strassen(A, B):
    # Base case for recursion: Matrix order n = 1 or 0
    if len(A) <= 2:
        return multiply_matrices(A, B)

    # Divide matrices into four submatrices each of size n/2 x n/2
    n = len(A)
    mid = n // 2
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]

    B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
    B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
    B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
    B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]

    # Calculate intermediate matrices p1 to p7 using Strassen algorithm recursively
    p1 = strassen(add_matrices(A11, A22), add_matrices(B11, B22))
    p2 = strassen(add_matrices(A21, A22), B11)
    p3 = strassen(A11, subtract_matrices(B12, B22))
    p4 = strassen(A22, subtract_matrices(B21, B11))
    p5 = strassen(add_matrices(A11, A12), B22)
    p6 = strassen(subtract_matrices(A21, A11), add_matrices(B11, B12))
    p7 = strassen(subtract_matrices(A12, A22), add_matrices(B21, B22))

    # Calculate C11, C12, C21, and C22 using intermediate results
    C11 = add_matrices(subtract_matrices(add_matrices(p1, p4), p5), p7)
    C12 = add_matrices(p3, p5)
    C21 = add_matrices(p2, p4)
    C22 = add_matrices(subtract_matrices(add_matrices(p1, p3), p2), p6)

    # Combine submatrices to form the final result
    C = []
    for i in range(mid):
        row = []
        row.extend(C11[i])
        row.extend(C12[i])
        C.append(row)
    for i in range(mid):
        row = []
        row.extend(C21[i])
        row.extend(C22[i])
        C.append(row)

    return C


def add_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if rows_A != rows_B or cols_A != cols_B:
        raise Exception("Incompatible matrix sizes for addition")

    return [[A[i][j] + B[i][j] for j in range(cols_A)] for i in range(rows_A)]

def subtract_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if rows_A != rows_B or cols_A != cols_B:
        raise Exception("Incompatible matrix sizes for subtraction")

    return [[A[i][j] - B[i][j] for j in range(cols_A)] for i in range(rows_A)]


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

