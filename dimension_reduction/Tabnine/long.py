from typing import Union
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd


class PrincipalComponentAnalysis:
    """
    A class for performing Principal Component Analysis (PCA) on data.

    Parameters
    ----------
    n_components : int, optional
        The number of principal components to keep, by default None (all components)
    decomposition_method : str, optional
        The method used for decomposing the data, either 'eigen' or 'svd', by default 'eigen'

    Attributes
    ----------
    components_ : numpy.ndarray
        The principal components
    explained_variance_ratio_ : numpy.ndarray
        The explained variance ratio for each principal component
    mean_ : numpy.ndarray
        The mean of the training data
    n_components_ : int
        The number of principal components used

    Methods
    -------
    fit(X: Union[numpy.ndarray, pandas.DataFrame, list])
        Fits the PCA model to the training data
    fit_transform(X: Union[numpy.ndarray, pandas.DataFrame, list], n_components: int = None)
        Fits the PCA model to the training data and transforms the data
    transform(X: Union[numpy.ndarray, pandas.DataFrame, list], n_components: int = None)
        Transforms the data using the fitted PCA model
    """

    def __init__(self, n_components: int = None, decomposition_method: str = 'eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method

    def fit(self, X: Union[np.ndarray, pd.DataFrame, list]):
        """
        Fits the PCA model to the training data.

        Parameters
        ----------
        X : Union[numpy.ndarray, pandas.DataFrame, list]
            The training data

        Returns
        -------
        self : PrincipalComponentAnalysis
            Returns the instance itself
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        self.mean_ = X.mean(axis=0)
        if self.decomposition_method == 'eigen':
            U, S, V = np.linalg.svd(X - self.mean_, full_matrices=False)
            self.components_ = V.T[:self.n_components]
            self.explained_variance_ratio_ = S[:self.n_components] / S.sum()
        elif self.decomposition_method == 'svd':
            U, S, V = randomized_svd(X - self.mean_, n_components=self.n_components, n_iter=7, random_state=0)
            self.components_ = V
            self.explained_variance_ratio_ = S / S.sum()
        else:
            raise ValueError(f"Invalid decomposition method: {self.decomposition_method}. "
                             f"Valid methods are 'eigen' and 'svd'.")
        self.n_components_ = self.components_.shape[0]
        return self

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame, list], n_components: int = None):
        """
        Fits the PCA model to the training data and transforms the data.

        Parameters
        ----------
        X : Union[numpy.ndarray, pandas.DataFrame, list]
            The training data
        n_components : int, optional
            The number of principal components to use, by default None (all components)

        Returns
        -------
        numpy.ndarray
            The transformed data
        """
        self.fit(X)
        return self.transform(X, n_components)

    def transform(self, X: Union[np.ndarray, pd.DataFrame, list], n_components: int = None):
        """
        Transforms the data using the fitted PCA model.

        Parameters
        ----------
        X : Union[numpy.ndarray, pandas.DataFrame, list]
            The data to transform
        n_components : int, optional
            The number of principal components to use, by default None (all components)

        Returns
        -------
        numpy.ndarray
            The transformed data
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        if n_components is None:
            n_components = self.n_components
        if self.decomposition_method == 'eigen':
            return X @ self.components_.T[:n_components] + self.mean_
        elif self.decomposition_method == 'svd':
            return U @ np.diag(S[:n_components])
        else:
            raise ValueError(f"Invalid decomposition method: {self.decomposition_method}. "
                             f"Valid methods are 'eigen' and 'svd'.")

    def __repr__(self):
        return (f"PrincipalComponentAnalysis(n_components={self.n_components}, "
                f"decomposition_method={self.decomposition_method})")