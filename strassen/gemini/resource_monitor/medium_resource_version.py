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


def is_power_of_2(n: int) -> bool:
  """Checks if a number is a power of 2."""
  return n != 0 and (n & (n - 1) == 0)


def is_square_matrix(matrix: np.ndarray) -> bool:
  """Checks if a matrix is square (equal rows and columns)."""
  return len(matrix) == len(matrix[0])


def check_matrix_compatibility(matrix1: np.ndarray, matrix2: np.ndarray) -> None:
  """
  Checks if two matrices are compatible for multiplication (equal inner dimensions).
  Raises ValueError if incompatible.
  """
  if not is_square_matrix(matrix1) or not is_square_matrix(matrix2):
    raise ValueError("Matrices must be square.")
  if matrix1.shape[0] != matrix2.shape[1]:
    raise ValueError("Incompatible matrix dimensions for multiplication.")


def pad_matrix(matrix: np.ndarray, size: int) -> np.ndarray:
  """Pads a matrix with zeros to reach the desired size (power of 2)."""
  pad_width = ((size - matrix.shape[0]) // 2, (size - matrix.shape[1]) // 2)
  return np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=0)


def strassen_multiply(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
  """
  Performs matrix multiplication using Strassen's algorithm.

  Args:
      matrix1: First input matrix.
      matrix2: Second input matrix.

  Returns:
      The product of the two matrices.
  """
  check_matrix_compatibility(matrix1, matrix2)

  # Base case: Use standard multiplication for small matrices
  n = matrix1.shape[0]
  if n <= 2:
    return np.dot(matrix1, matrix2)

  # Ensure matrices are power of 2 for sub-divisions
  if not is_power_of_2(n):
    new_size = 1 << (n.bit_length() - 1)
    matrix1 = pad_matrix(matrix1, new_size)
    matrix2 = pad_matrix(matrix2, new_size)

  # Subdivide matrices into sub-matrices
  a11, a12, a21, a22 = matrix1[:n // 2, :n // 2], matrix1[:n // 2, n // 2:], \
                         matrix1[n // 2:, :n // 2], matrix1[n // 2:, n // 2:]
  b11, b12, b21, b22 = matrix2[:n // 2, :n // 2], matrix2[:n // 2, n // 2:], \
                         matrix2[n // 2:, :n // 2], matrix2[n // 2:, n // 2:]

  # Recursive calls to calculate sub-problems
  p1 = strassen_multiply(a11 + a22, b11 + b22)
  p2 = strassen_multiply(a21 + a22, b11)
  p3 = strassen_multiply(a11, b12 - b22)
  p4 = strassen_multiply(a22, b21 - b11)
  p5 = strassen_multiply(a11 + a12, b22)
  p6 = strassen_multiply(a21 - a11, b11 + b12)
  p7 = strassen_multiply(a12 - a22, b21 + b22)

  # Combine sub-problems to form the result
  c11 = p1 + p4 - p5 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 + p6

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

