import numpy as np
import pandas as pd

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) is a dimensionality reduction technique that
    transforms a set of correlated variables into a set of linearly uncorrelated variables,
    called principal components, that capture the most important features of the data.
    The first principal component has the largest possible variance, and each subsequent
    component has the highest variance possible while being orthogonal to the previous
    components.

    This class implements PCA from scratch and provides methods for fitting the data,
    transforming the data, and fitting and transforming the data in one step.
    """

    def __init__(self, n_components=None, decomposition_method='eigen'):
        """
        Initializes the PrincipalComponentAnalysis class.

        Parameters:
            n_components (int, optional): The number of components to keep. If not specified,
                all components are kept. Default is None.
            decomposition_method (str, optional): The method used for decomposition.
                Options are 'eigen' or 'svd'. If not specified, 'eigen' is used. Default is 'eigen'.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method

    def fit(self, X):
        """
        Fits the data to the model.

        Parameters:
            X (numpy array, pandas dataframe, or list of lists): The data to fit.

        Returns:
            None

        Notes:
            Computes the eigen values and eigen vectors (components) for the inputted data.
            Stores the top n_components eigen values and eigen vectors on the object.
            Computes the explained variance ratio and cumulative sum of explained variance ratio
            for each component and stores it on the object.
        """
        if self.decomposition_method == 'eigen':
            # Eigenvalue decomposition
            eigen_vals, eigen_vecs = np.linalg.eig(X.cov())
            self.eigen_vals_ = eigen_vals
            self.eigen_vecs_ = eigen_vecs
        elif self.decomposition_method == 'svd':
            # Singular value decomposition
            svd = np.linalg.svd(X)
            self.eigen_vals_ = svd[0]
            self.eigen_vecs_ = svd[1]
        else:
            raise ValueError("Invalid decomposition method")

        # Select top n_components
        if self.n_components is not None:
            self.eigen_vals_ = self.eigen_vals_[:self.n_components]
            self.eigen_vecs_ = self.eigen_vecs_[:, :self.n_components]

        # Compute explained variance ratio and cumulative sum of explained variance ratio
        self.explained_variance_ratio_ = np.zeros(self.n_components)
        self.cumulative_explained_variance_ratio_ = np.zeros(self.n_components)
        for i in range(self.n_components):
            var = np.sum(self.eigen_vals_[i] ** 2)
            self.explained_variance_ratio_[i] = var / (var + np.sum(self.eigen_vals_[:i] ** 2))
            self.cumulative_explained_variance_ratio_[i] = self.explained_variance_ratio_[i] + np.sum(self.explained_variance_ratio_[:i])

    def transform(self, X):
        """
        Transforms the data onto the principal components.

        Parameters:
            X (numpy array, pandas dataframe, or list of lists): The data to transform.

        Returns:
            numpy array: The transformed data.

        Notes:
            Projects the inputted data onto the principal components.
            If n_components is not specified, transforms all components.
        """
        if self.n_components is None:
            n_components = self.n_components
        else:
            n_components = self.n_components_

        # Compute transformation matrix
        transformation_matrix = self.eigen_vecs_[:, :n_components]

        # Transform data
        transformed_data = np.dot(X, transformation_matrix)

        return transformed_data

    def fit_transform(self, X):
        """
        Fits the data and transforms it onto the principal components.

        Parameters:
            X (numpy array, pandas dataframe, or list of lists): The data to fit and transform.

        Returns:
            numpy array: The transformed data.

        Notes:
            Calls fit and transform methods.
            If n_components is not specified, transforms all components.
        """
        self.fit(X)
        return self.transform(X)
