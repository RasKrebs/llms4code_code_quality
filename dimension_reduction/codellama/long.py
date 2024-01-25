class PrincipalComponentAnalysis:
    """
    A class for performing Principal Component Analysis (PCA) on a dataset.

    Attributes:
        n_components: The number of components to keep in the PCA decomposition. If not specified, defaults to all components.
        decomposition_method: The method used for computing the eigen values and eigen vectors. Can be either 'eigen' or 'svd'. If not specified, defaults to 'eigen'.
        explained_variance: A list of floats representing the explained variance ratio for each component.
        cumulative_explained_variance: A list of floats representing the cumulative sum of the explained variance ratios for each component.
    """

    def __init__(self, n_components=None, decomposition_method='eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.explained_variance = []
        self.cumulative_explained_variance = []

    def fit(self, X):
        """
        Computes the eigen values and eigen vectors for the inputted data.

        Parameters:
            X (numpy array, pandas dataframe or list of lists): The input data to perform PCA on.

        Returns:
            A numpy array representing the transformed data.
        """
        if self.decomposition_method == 'eigen':
            eigen_values, eigen_vectors = np.linalg.eig(X.T @ X)
        elif self.decomposition_method == 'svd':
            _, _, vt = np.linalg.svd(X.T @ X)
            eigen_values = np.diag(np.dot(vt, np.diag(eigen_values)))
            eigen_vectors = vt
        else:
            raise ValueError("Invalid decomposition method")

        if self.n_components is not None and self.n_components < eigen_values.shape[0]:
            eigen_values = eigen_values[:self.n_components]
            eigen_vectors = eigen_vectors[:, :self.n_components]

        self.explained_variance = np.sum(eigen_values) / len(X)
        self.cumulative_explained_variance = np.cumsum(eigen_values) / len(X)

        return X @ eigen_vectors

    def transform(self, X, n_components=None):
        """
        Projects the input data onto the principal components.

        Parameters:
            X (numpy array, pandas dataframe or list of lists): The input data to perform PCA on.
            n_components (int): The number of components to project onto. If not specified, defaults to all components.

        Returns:
            A numpy array representing the transformed data.
        """
        if n_components is None:
            n_components = self.n_components

        return X @ self.eigenvectors[:, :n_components]

    def fit_transform(self, X, n_components=None):
        """
        Performs PCA on the input data and projects it onto the principal components.

        Parameters:
            X (numpy array, pandas dataframe or list of lists): The input data to perform PCA on.
            n_components (int): The number of components to project onto. If not specified, defaults to all components.

        Returns:
            A numpy array representing the transformed data.
        """
        self.fit(X)
        return self.transform(X, n_components=n_components)