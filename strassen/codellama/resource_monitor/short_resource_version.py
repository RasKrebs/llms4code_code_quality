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



import math

# Function to add two square matrices of the same size
def matrix_add(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]

    return C

# Function to subtract two square matrices of the same size
def matrix_subtract(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]

    return C

# Function to multiply two square matrices of the same size using Strassen's Algorithm
def strassen_multiply(A, B):
    n = len(A)

    # Base case: 1x1 matrix
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # Subdivide matrices into quadrants (n/2 x n/2 matrices)
    A11 = [[A[i][j] for j in range(n//2)] for i in range(n//2)]
    A12 = [[A[i][j] for j in range(n//2, n)] for i in range(n//2)]
    A21 = [[A[i][j] for j in range(n//2)] for i in range(n//2, n)]
    A22 = [[A[i][j] for j in range(n//2, n)] for i in range(n//2, n)]

    B11 = [[B[i][j] for j in range(n//2)] for i in range(n//2)]
    B12 = [[B[i][j] for j in range(n//2, n)] for i in range(n//2)]
    B21 = [[B[i][j] for j in range(n//2)] for i in range(n//2, n)]
    B22 = [[B[i][j] for j in range(n//2, n)] for i in range(n//2, n)]

    # Calculate intermediate matrices
    M1 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen_multiply(matrix_add(A21, A22), B11)
    M3 = strassen_multiply(A11, matrix_subtract(B12, B22))
    M4 = strassen_multiply(A22, matrix_subtract(B21, B11))
    M5 = strassen_multiply(matrix_add(A11, A12), B22)
    M6 = strassen_multiply(matrix_subtract(A21, A11), matrix_add(B11, B12))
    M7 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22))

    # Calculate quadrants of the final matrix C
    C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_subtract(matrix_subtract(matrix_add(M1, M3), M2), M6)

    # Combine quadrants into a single matrix by concatenating lists (rows)
    C = C11 + C21 + C12 + C22
    return [item for sublist in C for item in sublist]
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

