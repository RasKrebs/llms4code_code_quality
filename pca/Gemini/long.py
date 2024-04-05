import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) class for dimensionality reduction.

    This class performs PCA on a dataset using eigenvalue decomposition.

    Attributes:
        n_components (int): The number of principal components to use. Defaults to all.
        components_ (np.ndarray): The eigenvectors representing the principal components.
        explained_variance_ratio_ (np.ndarray): The explained variance ratio for each component.
        explained_variance_ (np.ndarray): The explained variance for each component.
        cumulative_explained_variance_ratio_ (np.ndarray): 
            The cumulative explained variance ratio for each component.
    """

    def __init__(self, n_components: int = None) -> None:
        """
        Initializes the PCA object.

        Args:
            n_components (int, optional): The number of principal components to use. 
                Defaults to all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.cumulative_explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model to the data X.

        This method computes the eigen values, eigen vectors, explained variance ratio,
        and cumulative explained variance ratio for the data.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
        """

        # Handle different data types by converting to float
        X = X.astype(np.float64)

        # Center the data by subtracting the mean
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # Compute the covariance matrix
        covariance = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]

        # Select the desired number of components
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        components = eigenvectors[:, : self.n_components]

        # Calculate explained variance ratio and cumulative sum
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues[: self.n_components]
        explained_variance_ratio = explained_variance / total_variance
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

        # Store fit results
        self.components_ = components
        self.explained_variance_ratio_ = explained_variance_ratio
        self.explained_variance_ = explained_variance
        self.cumulative_explained_variance_ratio_ = cumulative_explained_variance_ratio

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data X onto the principal components.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data of shape (n_samples, n_components).
        """

        # Check if PCA is fitted
        if self.components_ is None:
            raise ValueError("PCA object is not fitted. Call fit(X) before transform.")

        # Handle different data types by converting to float
        X = X.astype(np.float64)

        # Center the data using the mean from fit
        mean = getattr(self, "mean_", None)  # Use stored mean if available
        if mean is None:
            raise ValueError(
                "PCA object doesn't have mean information. Call fit(X) with the same data used for fitting."
            )
        X_centered = X - mean

        # Project data onto components
        return X_centered @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
            """
            Fits the PCA model to the data X and then transforms it.

            This method is equivalent to calling fit(X) followed by transform(X).

            Args:
                X (np.ndarray): The input data of shape (n_samples, n_features).

            Returns:
                np.ndarray: The transformed data of shape (n_samples, n_components).
            """

            # Call fit method to compute necessary components
            self.fit(X)

            # Call transform method to project data onto components
            return self.transform(X)
