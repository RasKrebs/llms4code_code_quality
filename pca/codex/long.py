import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implemented using eigenvalue decomposition.
    """
    def __init__(self, n_components: int = None):
        """
        Initialize the PCA object.

        :param n_components: Number of components to use. If not specified, all components are used.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the PCA model with the input data X.

        :param X: Input data.
        """
        # Convert input data to numpy array if it's not
        X = np.array(X, dtype=float)

        # Compute the mean
        self.mean = np.mean(X, axis=0)

        # Center the data
        X = X - self.mean

        # Compute the covariance matrix
        cov = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Use all components if n_components is not specified
        if self.n_components is None:
            self.n_components = len(eigenvalues)

        # Select the first n_components eigenvectors (components)
        self.components = eigenvectors[:, :self.n_components]

        # Compute explained variance and explained variance ratio
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        :param X: Input data.
        :return: Transformed data.
        """
        # Convert input data to numpy array if it's not
        X = np.array(X, dtype=float)

        # Center the data
        X = X - self.mean

        # Project data
        return np.dot(X, self.components)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        :param X: Input data.
        :return: Transformed data.
        """
        self.fit(X)
        return self.transform(X)