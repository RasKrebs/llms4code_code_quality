import numpy as np

def pca_from_eigen_decomposition(X):
    """
    Perform PCA from eigen decomposition on a dataset X.

    Args:
        X (numpy.ndarray): A dataset of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The transformed dataset of shape (n_samples, n_components)
    """
    # Center the dataset
    X_mean = X.mean(axis=0)
    X = X - X_mean

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    cov_mat = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    # Sort the eigenvalues and eigenvectors in decreasing order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep the first n_components eigenvalues and eigenvectors
    n_components = len(eigenvalues)
    transformed_X = eigenvectors[:, :n_components].T @ (X.T - X_mean)

    return transformed_X

def pca_from_singular_value_decomposition(X):
    """
    Perform PCA from singular value decomposition on a dataset X.

    Args:
        X (numpy.ndarray): A dataset of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The transformed dataset of shape (n_samples, n_components)
    """
    # Center the dataset
    X_mean = X.mean(axis=0)
    X = X - X_mean

    # Compute the SVD of the centered dataset
    U, S, Vt = np.linalg.svd(X)

    # Sort the singular values in decreasing order
    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:, idx]
    Vt = Vt[idx]

    # Keep the first n_components singular values
    n_components = len(S)
    transformed_X = U @ np.diag(S[:n_components]) @ Vt

    return transformed_X