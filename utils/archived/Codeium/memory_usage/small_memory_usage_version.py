from memory_profiler import profile
import numpy as np

class PCA:
    @profile
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    @profile
    def fit(self, X, method='eigen_decomposition'):
        # Calculate mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        if method == 'eigen_decomposition':
            # Using eigen decomposition
            cov_matrix = np.cov(X_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:,idx]
            self.components = eigenvectors[:,:self.n_components]
        elif method == 'singular_value_decomposition':
            # Using singular value decomposition
            U, s, Vt = np.linalg.svd(X_centered)
            self.components = Vt.T[:,:self.n_components]
        else:
            raise ValueError("Invalid method. Please choose either 'eigen_decomposition' or 'singular_value_decomposition'.")
    
    @profile
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PCA(10)
    pca.fit(data)
    _ = pca.transform(data)
    
