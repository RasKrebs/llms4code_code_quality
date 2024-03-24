class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA)

    Attributes:
        n_components (int): Number of principal components to use. If not
            specified, all components are used.
        decomposition_method (str): Method for computing eigenvalues and
            eigenvectors. Can be 'eigen' or 'svd'. If not specified, 'eigen' is
            used.
        eigen_values_ (ndarray): Eigenvalues of the data.
        eigen_vectors_ (ndarray): Eigenvectors of the data.
        explained_variance_ratio_ (ndarray): Ratio of the variance explained
            by each principal component.
        cumulative_explained_variance_ratio_ (ndarray): Cumulative ratio of
            the variance explained by each principal component.
    """

    def __init__(self, n_components=None, decomposition_method='eigen'):
        """
        Initialize a PrincipalComponentAnalysis object.

        Args:
            n_components (int, optional): Number of principal components to use.
                If not specified, all components are used.
            decomposition_method (str, optional): Method for computing
                eigenvalues and eigenvectors. Can be 'eigen' or 'svd'. If not
                specified, 'eigen' is used.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method

    def fit(self, X):
        """
        Fit the model to the data.

        Args:
            X (ndarray or DataFrame): Data to fit the model to. Can be a numpy
                array, pandas DataFrame, or a list of lists.

        Raises:
            ValueError: If the input data is not a numpy array, pandas DataFrame,
                or a list of lists.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not isinstance(X, (np.ndarray, list)):
            raise ValueError('Input data must be a numpy array, pandas '
                             'DataFrame, or a list of lists')

        # Compute eigenvalues and eigenvectors
        if self.decomposition_method == 'eigen':
            self._fit_eigen(X)
        elif self.decomposition_method == 'svd':
            self._fit_svd(X)
        else:
            raise ValueError(f'Invalid decomposition method: {self.decomposition_method}')

    def transform(self, X):
        """
        Project the data onto the principal components.

        Args:
            X (ndarray or DataFrame): Data to project. Can be a numpy array,
                pandas DataFrame, or a list of lists.

        Returns:
            ndarray: Projected data. Shape is (n_samples, n_components) if
                n_components is not specified, or (n_samples, n_components) if
                n_components is specified.

        Raises:
            ValueError: If the model is not fitted or if the input data is not a
                numpy array, pandas DataFrame, or a list of lists.
        """
        if not self.is_fitted_:
            raise ValueError('Model must be fitted before transforming data')

        if isinstance(X, pd.DataFrame):
            X = X.values

        if not isinstance(X, (np.ndarray, list)):
            raise ValueError('Input data must be a numpy array, pandas '
                             'DataFrame, or a list of lists')

        # Project data onto principal components
        if self.decomposition_method == 'eigen':
            return self._transform_eigen(X)
        elif self.decomposition_method == 'svd':
            return self._transform_svd(X)
        else:
            raise ValueError(f'Invalid decomposition method: {self.decomposition_method}')

    def fit_transform(self, X):
        """
        Fit the model to the data and project the data onto the principal
        components.

        Args:
            X (ndarray or DataFrame): Data to fit the model to and project. Can
                be a numpy array, pandas DataFrame, or a list of lists.

        Returns:
            ndarray: Projected data. Shape is (n_samples, n_components) if
                n_components is not specified, or (n_samples, n_components) if
                n_components is specified.

        Raises:
            ValueError: If the input data is not a numpy array, pandas DataFrame,
                or a list of lists.
        """
        self.fit(X)
        return self.transform(X)

    def _fit_eigen(self, X):
        """
        Compute eigenvalues and eigenvectors using the eigendecomposition of
        X.

        Args:
            X (ndarray): Data to fit the model to.
        """
        # Compute eigenvalues and eigenvectors
        self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(np.cov(X.T))

        # Sort eigenvalues and eigenvectors by decreasing eigenvalue
        idx = self.eigen_values_.argsort()[::-1]
        self.eigen_values_ = self.eigen_values_[idx]
        self.eigen_vectors_ = self.eigen_vectors_[:, idx]

        # Store explained variance ratio and cumulative explained variance ratio
        self.explained_variance_ratio_ = self.eigen_values_ ** 2 / np.sum(self.eigen_values_ ** 2)
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def _fit_svd(self, X):
        """
        Compute eigenvalues and eigenvectors using the singular value
        decomposition of X.

        Args:
            X (ndarray): Data to fit the model to.
        """
        # Compute singular value decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Sort singular values by decreasing magnitude
        idx = S.argsort()[::-1]
        S = S[idx]
        U = U[:, idx]
        Vt = Vt[idx]

        # Store explained variance ratio and cumulative explained variance ratio
        self.explained_variance_ratio_ = S ** 2 / np.sum(S ** 2)
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def _transform_eigen(self, X):
        """
        Project the data onto the principal components using the eigendecomposition
        of X.

        Args:
            X (ndarray): Data to project.

        Returns:
            ndarray: Projected data. Shape is (n_samples, n_components) if
                n_components is not specified, or (n_samples, n_components) if
                n_components is specified.
        """
        # Project data onto principal components
        if self.n_components is None:
            return X @ self.eigen_vectors_
        else:
            return X @ self.eigen_vectors_[:, :self.n_components]

    def _transform_svd(self, X):
        """
        Project the data onto the principal components using the singular value
        decomposition of X.

        Args:
            X (ndarray): Data to project.

        Returns:
            ndarray: Projected data. Shape is (n_samples, n_components) if
                n_components is not specified, or (n_samples, n_components) if
                n_components is specified.
        """
        # Project data onto principal components
        if self.n_components is None:
            return X @ self.Vt_
        else:
            return X @ self.Vt_[:, :self.n_components]

    @property
    def is_fitted_(self):
        """
        Returns:
            bool: Whether or not the model is fitted.
        """
        return self.eigen_values_ is not None

    @property
    def n_features_(self):
        """
        Returns:
            int: Number of features in the original data.
        """
        if self.is_fitted_:
            return self.eigen_vectors_.shape[1]
        else:
            return None