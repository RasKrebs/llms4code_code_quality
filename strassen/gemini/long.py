import numpy as np


def is_power_of_2(n: int) -> bool:
  """Checks if a number is a power of 2."""
  return n != 0 and (n & (n - 1) == 0)


def is_square_matrix(matrix: list[list]) -> bool:
  """Checks if a list of lists represents a square matrix."""
  return all(len(row) == len(matrix) for row in matrix)


def check_matrix_dimensions(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
  """
  Checks if the input matrices have valid dimensions for multiplication.

  Raises:
      ValueError: If matrices are not square, have different dimensions,
                  or their dimensions are not powers of 2.
  """
  if not is_square_matrix(matrix_a.tolist()) or not is_square_matrix(matrix_b.tolist()):
    raise ValueError("Input matrices must be square.")
  if matrix_a.shape[0] != matrix_b.shape[0]:
    raise ValueError("Matrices must have the same dimensions for multiplication.")
  if not is_power_of_2(matrix_a.shape[0]):
    raise ValueError("Matrix dimensions must be powers of 2 for Strassen's algorithm.")


def add_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
  """Adds two matrices element-wise."""
  return np.add(matrix_a, matrix_b)


def subtract_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
  """Subtracts two matrices element-wise."""
  return np.subtract(matrix_a, matrix_b)


def strassen_multiply(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
  """
  Multiplies two matrices using Strassen's algorithm.

  Args:
      matrix_a: The first square matrix.
      matrix_b: The second square matrix.

  Returns:
      The product of the two matrices.

  Raises:
      ValueError: If the input matrices are not valid for Strassen's algorithm.
  """
  check_matrix_dimensions(matrix_a, matrix_b)

  # Base case for small matrices
  n = matrix_a.shape[0]
  if n == 1:
    return np.array([[matrix_a[0, 0] * matrix_b[0, 0]]])

  # Divide matrices into sub-matrices
  half_size = n // 2

  sub_a11 = matrix_a[:half_size, :half_size]
  sub_a12 = matrix_a[:half_size, half_size:]
  sub_a21 = matrix_a[half_size:, :half_size]
  sub_a22 = matrix_a[half_size:, half_size:]

  sub_b11 = matrix_b[:half_size, :half_size]
  sub_b12 = matrix_b[:half_size, half_size:]
  sub_b21 = matrix_b[half_size:, :half_size]
  sub_b22 = matrix_b[half_size:, half_size:]

  # Recursive calls to calculate sub-problems
  p1 = strassen_multiply(add_matrices(sub_a11, sub_a22), add_matrices(sub_b11, sub_b22))
  p2 = strassen_multiply(sub_a21, add_matrices(sub_b11, sub_b12))
  p3 = strassen_multiply(add_matrices(sub_a11, sub_a12), sub_b22)
  p4 = strassen_multiply(sub_a22, subtract_matrices(sub_b21, sub_b11))
  p5 = strassen_multiply(add_matrices(sub_a11, sub_a21), sub_b22)
  p6 = strassen_multiply(subtract_matrices(sub_a12, sub_a22), add_matrices(sub_b11, sub_b21))
  p7 = strassen_multiply(subtract_matrices(sub_a11, sub_a21), add_matrices(sub_b21, sub_b22))

  # Combining sub-problems to form the result matrix
  c11 = add_matrices(subtract_matrices(add_matrices(p1, p4), p5), p7)
  c12 = add_matrices(p3, p5)
  c21 = add_matrices(p2, p4)
  c22 = add_matrices(subtract_matrices(add_matrices(p1, p3), p2), p6)

  # Form the resulting product matrix
  result = np.block([[c11, c12], [c21, c22]])

  return result

