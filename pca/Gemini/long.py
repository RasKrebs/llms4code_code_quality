import numpy as np
import pandas as pd

class PrincipalComponentAnalysis:
    """
    Implements Principal Component Analysis (PCA) from scratch.

    Parameters:
    n_components: (int, optional)
        The number of principal components to keep. If not specified, keeps all components.
    decomposition_method: (str, optional)
        The decomposition method to use. Can be either 'eigen' or 'svd'. If not specified, defaults to 'eigen'.
    """

    def __init__(self, n_components=None, decomposition_method='eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method

        # Eigen values and eigen vectors of the covariance matrix
        self.eigenvalues = None
        self.eigenvectors = None

        # Explained variance ratio and cumulative sum of explained variance ratio
        self.explained_variance_ratio = None
        self.cumulative_explained_variance_ratio = None

    def fit(self, X):
        """
        Fits the PCA model to the data.

        Parameters:
        X: (numpy.ndarray, pandas.DataFrame, list)
            The data to fit the model to. Can be a numpy array, pandas DataFrame, or a list of lists.
        """

        # Check if X is a valid data format
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError('X must be a numpy.ndarray, pandas.DataFrame, or a list of lists.')

        # Check if X is a 2D array
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError('X must be a 2D array.')

        # Check if X contains numerical data
        for row in X:
            if not all(isinstance(x, (int, float)) for x in row):
                raise ValueError('X must contain numerical data.')

        # Standardize the data
        X = StandardScaler().fit_transform(X)

        # Compute the covariance matrix
        covariance_matrix = np.cov(X.T)

        # Calculate the eigen values and eigen vectors
        if self.decomposition_method == 'eigen':
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        elif self.decomposition_method == 'svd':
            U, S, Vh = np.linalg.svd(covariance_matrix)
            eigen_values = S.astype(float)
            eigen_vectors = Vh.T.astype(float)
        else:
            raise ValueError('Invalid decomposition_method: {}.'.format(self.decomposition_method))

        # Sort eigen values and eigen vectors in descending order based on eigen values
        sorted_idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_idx]
        eigen_vectors = eigen_vectors[sorted_idx]

        # Keep only the top n_components
        if self.n_components is not None:
            eigen_values = eigen_values[:self.n_components]
            eigen_vectors = eigen_vectors[:self.n_components]

        # Compute explained variance ratio and cumulative sum of explained variance ratio
        explained_variance_ratio = eigen_values / np.sum(eigen_values)
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

        # Store the computed values
        self.eigenvalues = eigen_values
        self.eigenvectors = eigen_vectors
        self.explained_variance_ratio = explained_variance_ratio
        self.cumulative_explained_variance_ratio = cumulative_explained_variance_ratio
        
    def transform(self, X):
        """
        Projects the data onto the principal components.
    
        Parameters:
        X: (numpy.ndarray, pandas.DataFrame, list)
            The data to transform. Can be a numpy array, pandas DataFrame, or a list of lists.
    
        Returns:
            (numpy.ndarray)
                The projected data.
        """
    
        # Check if X is a valid data format
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError('X must be a numpy.ndarray, pandas.DataFrame, or list of lists.')
    
        # Check if X is a 2D array
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError('X must be a 2D array.')
    
        # Check if X contains numerical data
        for row in X:
            if not all(isinstance(x, (int, float)) for x in row):
                raise ValueError('X must contain numerical data.')
    
        # Transform the data
        transformed_data = np.dot(X, self.eigenvectors.T)
    
        return transformed_data