from memory_profiler import profile
import numpy as np

class PCA:
    @profile
    def __init__(self, n_components, method='eigen'):
        self.n_components = n_components
        self.method = method
        self.components = None
        self.mean = None

    @profile
    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigen decomposition
        if self.method == 'eigen':
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # -> eigenvector v = [:,i] column vector, transpose for easier calculations
            # sort eigenvectors
            eigenvectors = eigenvectors.T
            idxs = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
            # store first n eigenvectors
            self.components = eigenvectors[0:self.n_components]

        # singular value decomposition
        elif self.method == 'svd':
            u, s, v = np.linalg.svd(X.T)
            self.components = v[0:self.n_components]

    @profile
    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PCA(n_components=10, method='eigen')
    pca.fit(data)
    _ = pca.transform(data)
    