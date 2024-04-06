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

def split(matrix):
    """Split a given matrix into quarters."""
    row, col = matrix.shape
    mid_row, mid_col = row//2, col//2

    return matrix[:mid_row, :mid_col], matrix[:mid_row, mid_col:], \
           matrix[mid_row:, :mid_col], matrix[mid_row:, mid_col:]

def strassen(A, B):
    """Multiply two matrices using Strassen's algorithm."""
    # Base case when size of the matrices is 1x1
    if len(A) == 1:
        return A * B

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(A)
    e, f, g, h = split(B)

    # Computing the 7 products: p1 to p7
    p1 = strassen(a, (f - h))
    p2 = strassen((a + b), h)
    p3 = strassen((c + d), e)
    p4 = strassen(d, (g - e))
    p5 = strassen((a + d), (e + h))
    p6 = strassen((b - d), (g + h))
    p7 = strassen((a - c), (e + f))

    # Computing the values of the 4 quadrants of the final matrix c
    c1 = p5 + p4 - p2 + p6
    c2 = p1 + p2
    c3 = p3 + p4
    c4 = p1 + p5 - p3 - p7

    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    top = np.hstack((c1, c2))
    bottom = np.hstack((c3, c4))
    return np.vstack((top, bottom))
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

