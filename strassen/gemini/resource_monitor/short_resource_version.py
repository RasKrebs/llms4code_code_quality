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



def strassen(A, B):
  """
  Performs matrix multiplication using the Strassen algorithm.

  Args:
      A: A square matrix (n x n).
      B: A square matrix (n x n).

  Returns:
      The resulting product matrix (n x n).
  """
  # Base case: Matrices of size 1x1
  n = len(A)
  if n == 1:
    return [[A[0][0] * B[0][0]]]

  # Subdivide matrices into quarters
  a11, a12, a21, a22 = split_matrix(A)
  b11, b12, b21, b22 = split_matrix(B)

  # Seven subproblems
  p1 = strassen(add(a11, a22), add(b11, b22))
  p2 = strassen(add(a21, a22), b11)
  p3 = strassen(a11, subtract(b12, b22))
  p4 = strassen(a22, subtract(b21, b11))
  p5 = strassen(add(a11, a12), b22)
  p6 = strassen(subtract(a21, a11), add(b11, b12))
  p7 = strassen(subtract(a12, a22), add(b21, b22))

  # Combine subproblems
  c11 = add(subtract(p1, p4), add(p5, p7))
  c12 = add(p3, p5)
  c21 = add(p2, p4)
  c22 = add(subtract(p1, p3), add(p6, p7))

  # Assemble resulting matrix
  return assemble_matrix(c11, c12, c21, c22)

def add(A, B):
  """
  Adds two matrices element-wise.
  """
  n = len(A)
  C = [[0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
    for j in range(n):
      C[i][j] = A[i][j] + B[i][j]
  return C

def subtract(A, B):
  """
  Subtracts two matrices element-wise.
  """
  return add(A, [[-x for x in row] for row in B])

def split_matrix(M):
  """
  Splits a square matrix into four sub-matrices.
  """
  n = len(M)
  half = n // 2
  return (M[:half], M[:half][half:], M[half:], M[half:][half:])

def assemble_matrix(c11, c12, c21, c22):
  """
  Assembles four sub-matrices into a resulting matrix.
  """
  n = len(c11)
  return [
      c11 + c12,
      c21 + c22,
  ]

# Example usage
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

C = strassen(A, B)
print(C)
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

