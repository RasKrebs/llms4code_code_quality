import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, n_components: int = None, decomposition_method: str = 'eigen'):
        """
        Initialize the Principal Component Analysis class.

        Args:
        - n_components (int, optional): Number of components to use. Defaults to None.
        - decomposition_method (str, optional): Decomposition method to use, either 'eigen' or 'svd'. Defaults to 'eigen'.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.components = None

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model according to the given training data.

        Args:
        - X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        # Calculate the mean of each feature
        mean = np.mean(X, axis=0)
        
        # Center the data
        centered_data = X - mean
        
        # Calculate the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        
        if self.decomposition_method == 'eigen':
            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            self.components = sorted_eigenvectors[:, :self.n_components]
        elif self.decomposition_method == 'svd':
            # Singular Value Decomposition (SVD)
            _, s, vh = np.linalg.svd(centered_data)
            self.components = vh[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Args:
        - X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
        - np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        # Center the data
        centered_data = X - np.mean(X, axis=0)
        
        # Apply the transformation
        transformed_X = np.dot(centered_data, self.components)
        return transformed_X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Args:
        - X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
        - np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)
