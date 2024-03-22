from memory_profiler import profile
import numpy as np

class PrincipalComponentAnalysis:
    """
    Implements the Principal Component Analysis (PCA) algorithm from scratch.
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to retain. If not specified, all components are retained.
    decomposition_method : str, optional
        Method used for decomposition. Can be either 'eigen' or 'svd'. If not specified, 'eigen' is used.
    
    Methods
    -------
    fit(X)
        Fits the PCA model to the data.
    transform(X)
        Transforms the data into the principal components.
    fit_transform(X)
        Fits the PCA model to the data and transforms it into the principal components.
    """
    @profile
    def __init__(self, n_components=None, decomposition_method='eigen'):
        """
        Initializes the PCA model.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method
    @profile
    def fit(self, X):
        """
        Fits the PCA model to the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to fit the model to.
        """
        # Compute covariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        
        # Compute eigenvectors and eigenvalues
        if self.decomposition_method == 'eigen':
            eigenvectors, eigenvalues = np.linalg.eig(cov_matrix)
        else:
            eigenvectors, eigenvalues = np.linalg.svds(cov_matrix, k=.n_components)
        
        # Select top n_components eigenvectors and eigenvalues
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
            eigenvalues = eigenvalues[:self.n_components]
        
        # Normalize eigenvectors
        eigenvectors = eigenvectors / np.sqrt(eigenvalues)
        
        # Save eigenvectors and eigenvalues for transformation
        self.eigenvectors_ = eigenvectors
        self.eigenvalues_ = eigenvalues
    
    @profile
    def transform(self, X):
        """
        Transforms the data into the principal components.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
        """
        # Compute dot product of data and eigenvectors
        transformed_data = np.dot(X, self.eigenvectors_)
        
        return transformed_data
    
    @profile
    def fit_transform(self, X):
        """
        Fits the PCA model to the data and transforms it into the principal components.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to fit the model to and transform.
        """
        self.fit(X)
        return self.transform(X)

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PrincipalComponentAnalysis(n_components=5)

    # Fit and transform data
    _ = pca.fit_transform(data)