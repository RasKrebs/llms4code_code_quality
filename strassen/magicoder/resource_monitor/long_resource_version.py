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

def strassens_matrix_mul(a, b):
    """
    Perform matrix multiplication using Strassen's algorithm.

    :param a: first operand, numpy array or list of lists
    :type a: numpy.ndarray or list

    :param b: second operand, numpy array or list of lists
    :type b: numpy.ndarray or list

    :return: product of the two matrices
    :rtype: numpy.ndarray
    """
    # Check if a and b are valid matrices
    assert isinstance(a, (list, np.ndarray)), "First argument should be a matrix"
    assert isinstance(b, (list, np.ndarray)), "Second argument should be a matrix"

    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b

    # Check if they are valid square matrices
    assert len(set([len(row) for row in a])) == 1 and len(a[0]) == len(b), "Matrices can't be multiplied"

    n = len(a)
    mid = int(n/2)  # integer division to ensure we round down

    if n==1:   # base case
        return np.array([[a[0][0]*b[0][0]]])

    a11, a12, a21, a22 = a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]
    b11, b12, b21, b22 = b[:mid, :mid], b[:mid, mid:], b[mid:, :mid], b[mid:, mid:]

    # Strassen's algorithm step by step
    p1 = strassens_matrix_mul(a11 + a22, b11 + b22)
    p2 = strassens_matrix_mul(a21, b12 - b22)  # s is for subtraction operation
    p3 = strassens_matrix_mul(a11, b11 + b12)
    p4 = strassens_matrix_mul(a22, b21 - b11)
    p5 = strassens_matrix_mul(a11 + a12, b22)
    p6 = strassens_matrix_mul(a21 - a11, (b11 + b12))
    p7 = strassens_matrix_mul(a12 - a22, (b21 + b22))

    c11 = p1 + p4 - p5 + p7  # Calculate individual sub-matrices
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 + p3 - p2 + p6

    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))  # Combine sub-matrices to get the final result
def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim)) 
    B = np.random.randint(1, 10, size=(dim, dim))
    
    # Perform matrix multiplication using the Strassen algorithm
    C = strassens_matrix_mul(A, B)



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

