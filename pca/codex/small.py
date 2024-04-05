import numpy as np

def PCA(X, num_components):
    # 1. Standardize the dataset
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # 2. Compute the covariance matrix
    covariance_matrix = np.cov(X.T)
    
    # 3. Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # 4. Sort the eigenvalues and corresponding eigenvectors
    sorted_components = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_components]
    sorted_eigenvectors = eigenvectors[:,sorted_components]
    
    # 5. Select the first k eigenvectors
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    
    # 6. Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(),X.transpose()).transpose()

    return X_reduced