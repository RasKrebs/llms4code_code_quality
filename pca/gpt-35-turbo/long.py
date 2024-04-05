class PCA:
    def __init__(self, n_components=None):
        """
        Initialize PCA object with number of components.
        
        Parameters:
        n_components (int): Number of components to use. Default is None.
        """
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance_ratio = None
        
    def fit(self, X):
        """
        Fit the data and compute eigenvalues, eigenvectors, explained variance ratio and cumulative explained variance ratio.
        
        Parameters:
        X (numpy array): Input data
        
        Returns:
        None
        """
        # Ensure X is a numpy array
        if type(X) != np.ndarray:
            X = np.array(X)
        
        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        
        # Compute covariance matrix
        cov_mat = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        self.eigenvalues, self.components = np.linalg.eig(cov_mat)
        
        # Sort components based on eigenvalues
        idx = np.argsort(self.eigenvalues)[::-1]
        self.components = self.components[:, idx]
        self.eigenvalues = self.eigenvalues[idx]
        
        # Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        
        # Compute cumulative explained variance ratio
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)
        
    def transform(self, X):
        """
        Project input data onto the components.
        
        Parameters:
        X (numpy array): Input data
        
        Returns:
        X_projected (numpy array): Projected data
        """
        # Ensure X is a numpy array
        if type(X) != np.ndarray:
            X = np.array(X)
        
        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        
        # Project data onto components
        X_projected = np.dot(X_centered, self.components)
        
        return X_projected
    
    def fit_transform(self, X):
        """
        Fit the data and transform it using the computed components.
        
        Parameters:
        X (numpy array): Input data
        
        Returns:
        X_projected (numpy array): Projected data
        """
        self.fit(X)
        return self.transform(X)