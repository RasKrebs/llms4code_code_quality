from memory_profiler import profile
import numpy as np

@profile
def pca_eigen(X):
    """
    Perform Principal Component Analysis (PCA) on the input matrix X using Eigen decomposition.
    """
    # Compute covariance matrix
    cov = np.cov(X.T)
    
    # Compute eigenvectors and eigenvalues of covariance matrix
    eigenvectors, eigenvalues = np.linalg.eig(cov)
    
    # Sort eigenvectors and eigenvalues by magnitude of eigenvalues
    indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, indices]
    eigenvalues = eigenvalues[indices]
    
    # Select k eigenvectors corresponding to largest eigenvalues
    k = min(X.shape[1], 10)
    eigenvectors = eigenvectors[:, :k]
    eigenvalues = eigenvalues[:k]
    
    # Compute principal components
    principal_components = np.dot(X, eigenvectors)
    
    return principal_components, eigenvectors, eigenvalues
@profile
def pca_svd(X):
    """
    Perform Principal Component Analysis (PCA) on the input matrix X using Singular Value Decomposition (SVD).
    """
    # Compute SVD of X
    U, sigma, Vt = np.linalg.svd(X)
    
    # Select k singular values and corresponding singular vectors
    k = min(X.shape[1], 10)
    sigma_k = sigma[:k]
    U_k = U[:, :k]
    Vt_k = Vt[:k]
    
    # Compute principal components
    principal_components = np.dot(U_k, np.dot(sigma_k, Vt_k))
    
    return principal_components, U_k, sigma_k, Vt_k

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = pca_eigen(data)