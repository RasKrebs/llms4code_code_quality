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


def is_power_of_2(n):
  """Checks if a number is a power of 2."""
  return n != 0 and (n & (n - 1) == 0)


def check_matrix(matrix):
  """
  Checks if the input is a valid square matrix.

  Args:
    matrix: The input matrix (list of lists, numpy array, or pandas dataframe)

  Raises:
    ValueError: If the input is not a list of lists, numpy array, or pandas dataframe.
    ValueError: If the matrix is not square.
    ValueError: If the matrix dimensions are not powers of 2.
  """
  if not isinstance(matrix, (list, np.ndarray, pd.DataFrame)):
    raise ValueError("Input must be a list of lists, numpy array, or pandas dataframe")

  rows, cols = len(matrix), len(matrix[0])
  if rows != cols:
    raise ValueError("Input matrix must be square")
  if not is_power_of_2(rows):
    raise ValueError("Matrix dimensions must be powers of 2")


def matrix_add(A, B):
  """
  Adds two matrices element-wise.

  Args:
    A: The first matrix.
    B: The second matrix.

  Returns:
    The sum of the two matrices.
  """
  return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matrix_sub(A, B):
  """
  Subtracts two matrices element-wise.

  Args:
    A: The first matrix.
    B: The second matrix.

  Returns:
    The difference of the two matrices.
  """
  return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def strassen(A, B):
  """
  Multiplies two matrices using the Strassen algorithm.

  Args:
    A: The first matrix.
    B: The second matrix.

  Returns:
    The product of the two matrices.

  Raises:
    ValueError: If the matrices have incompatible dimensions.
  """
  check_matrix(A)
  check_matrix(B)
  n = len(A)

  # Base case: use standard multiplication for small matrices
  if n == 1:
    return [[A[0][0] * B[0][0]]]

  # Split matrices into sub-matrices
  half = n // 2
  sub11, sub12, sub21, sub22 = A[:half], A[half:], B[:half], B[half:]
  sub31, sub32, sub41, sub42 = (
      A[:half],
      A[half:],
      B[:, :half],
      B[:, half:],
  )

  # Recursive calls to compute sub-products
  p1 = strassen(matrix_add(sub11, sub22), matrix_add(sub41, sub42))
  p2 = strassen(sub21, sub41)
  p3 = strassen(sub11, matrix_sub(sub31, sub32))
  p4 = strassen(sub22, sub32)
  p5 = strassen(matrix_add(sub11, sub12), sub42)
  p6 = strassen(matrix_sub(sub21, sub11), matrix_add(sub12, sub22))
  p7 = strassen(matrix_sub(sub31, sub12), matrix_add(sub32, sub42))

  # Combine sub-products to form the resulting matrix
  C11 = matrix_add(matrix_sub(p1, p4), matrix_add(p5, p7))
  C12 = matrix_add(p3, p5)
  C21 = matrix_add(p2, p4)
  C22 = matrix_add(matrix_sub(p1, p3), matrix_add(p6, p7))

  return np.block([[C11, C12], [C21, C22]])
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

