import numpy as np
from memory_profiler import profile


class PCA:
  def __init__(self, n_components):
    self.n_components = n_components

  @profile
  def fit(self, X):
    # Center data
    self.mean_ = np.mean(X, axis=0)
    X -= self.mean_
    
    # Calculate covariance matrix
    cov_mat = np.cov(X.T)
    
    # Perform eigendecomposition of covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    # Sort eigenvalues in descending order and select top n
    eigen_pairs = sorted(zip(eigen_vals, eigen_vecs), 
                         key=lambda x: x[0], reverse=True)
    self.components_ = np.array([eigen_pairs[i][1] for i in range(self.n_components)])

    return self

  @profile
  def transform(self, X):
    # Project data onto selected components
    projected = np.dot((X-self.mean_), self.components_.T)
    return projected
  

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PCA(n_components=10)

    # Fit and transform data
    _ = pca.fit(data).transform(data)
