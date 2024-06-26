import numpy as np

def pca(X, k):
  """
  Performs PCA on data matrix X and returns projected data and explained variance.

  Args:
    X: Data matrix (n x m).
    k: Number of principal components to retain.

  Returns:
    X_reduced: Projected data matrix (n x k).
    explained_variance: Explained variance ratio for each component.
  """
  # Center data (optional)
  X_centered = X - np.mean(X, axis=0)

  # Covariance matrix
  covariance = np.cov(X_centered.T)

  # Eigenvalue decomposition
  eigenvalues, eigenvectors = np.linalg.eig(covariance)

  # Sort eigenvectors by eigenvalues (descending)
  sorted_index = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sorted_index]
  eigenvectors = eigenvectors[:, sorted_index]

  # Select top k eigenvectors
  V_reduced = eigenvectors[:, :k]

  # Project data onto principal components
  X_reduced = X_centered @ V_reduced

  # Explained variance ratio
  explained_variance = eigenvalues / np.sum(eigenvalues)

  return X_reduced, explained_variance
