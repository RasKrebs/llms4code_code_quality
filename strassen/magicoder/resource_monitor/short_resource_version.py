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
from math import floor

def strassen_matrix_mult(A, B):
    def splitMatrix(mat):
        row, col = mat.shape
        return mat[:row//2, :col//2], mat[:row//2, col//2:], mat[row//2:, :col//2], mat[row//2:, col//2:]

    if len(A) == 1:
        return A * B

    a, b, c, d = splitMatrix(A)
    e, f, g, h = splitMatrix(B)

    p1 = strassen_matrix_mult(a, f - h)
    p2 = strassen_matrix_mult(a + b, h)
    p3 = strassen_matrix_mult(c + d, e)
    p4 = strassen_matrix_mult(d, g - e)
    p5 = strassen_matrix_mult(a + d, e + h)
    p6 = strassen_matrix_mult(b - d, g + h)
    p7 = strassen_matrix_mult(a - c, e + f)

    res11 = p5 + p4 - p2 + p6
    res12 = p1 + p2
    res21 = p3 + p4
    res22 = p1 + p5 - p3 - p7

    return np.vstack((np.hstack((res11, res12)), np.hstack((res21, res22))))

def execute():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Define the dimensions for the matrices
    dim = 64 
    
    # Generate random matrices A and B of size dim x dim
    A = np.random.randint(1, 10, size=(dim, dim))
    B = np.random.randint(1, 10, size=(dim, dim))
    
    # Perform matrix multiplication using the Strassen algorithm
    C = strassen_matrix_mult(A, B)



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

