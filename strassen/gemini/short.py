import numpy as np

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
