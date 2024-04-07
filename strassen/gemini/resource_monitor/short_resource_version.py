import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
import sys
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

def strassen(A, B):
  """
  Performs matrix multiplication using Strassen's algorithm.

  Args:
      A: A square NumPy matrix.
      B: A square NumPy matrix with the same dimension as A.

  Returns:
      The product of A and B as a NumPy matrix.
  """
  n = A.shape[0]

  # Base case: Matrices of size 1x1
  if n == 1:
    return A * B

  # Divide matrices into sub-matrices
  a11, a12, a21, a22 = A[:n // 2, :n // 2], A[:n // 2, n // 2:], A[n // 2:, :n // 2], A[n // 2:, n // 2:]
  b11, b12, b21, b22 = B[:n // 2, :n // 2], B[:n // 2, n // 2:], B[n // 2:, :n // 2], B[n // 2:, n // 2:]

  # Recursively compute sub-matrix products
  p1 = strassen(a11 + a22, b11 + b22)
  p2 = strassen(a21 + a22, b11)
  p3 = strassen(a11, b12 - b22)
  p4 = strassen(a22, b21 - b11)
  p5 = strassen(a11 + a12, b22)
  p6 = strassen(a21 - a11, b11 + b12)
  p7 = strassen(a12 - a22, b21 + b22)

  # Combine sub-matrix products
  c11 = p1 + p4 - p5 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 + p6

  # Assemble the result matrix
  return np.block([[c11, c12], [c21, c22]])

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

