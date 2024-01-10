### ADD SELFMADE SOURCE CODE AND NECESSARY IMPORTS
import numpy as np
import pandas as pd
from typing import Union


class PrincipalComponentAnalysis:
    def __init__(self, 
                 n_components:int = 0,
                 decomposition:str = 'eigen'):
        """
        Initializes the Principal Component Analysis model.

        Parameters:
        - n_components (int): The number of components to keep. Default is all.

        Methods:
        - fit(X): Applies PCA to the input data.
        - transform(X, n_components): Transforms the input data to the specified number of components using the fitted PCA model.
        - fit_transform(X): Applied pca to input, transforms to specified number of components and returns the reduced data.
        
        Attributes:
        - n_components (int): The number of components to keep.
        - components (None): Placeholder for the components.
        - mean (None): Placeholder for the sample mean.
        - decomposition (str): The type of decomposition to use. Either 'eigen' or 'svd'.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        
        assert decomposition in ['eigen', 'svd'], "Decomposition must be either 'eigen' or 'svd'"
        self.decomposition = decomposition
        
        print(f"Principal Component Analysis initialized with {self.decomposition} decomposition")
        

    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray, list]):
        """
        Applies PCA to the input data.

        Parameters:
            X (pandas.DataFrame, numpy.ndarray, list): The input data to fit the model on. 
            
        Raises:
            TypeError: If X is not a pandas DataFrame, numpy array, or list.

        Returns:
            None
        """
        
        # Handling different input types
        if type(X) == pd.DataFrame: X = X.values
        elif type(X) == list: X = np.array(X)
        elif type(X) == np.ndarray: pass
        else: raise TypeError("X must be a pandas dataframe, numpy array or list")
        
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Standard decomposition
        if self.decomposition == 'eigen':
            # 2. Compute Sample Covariance Matrix
            cov = 1/(len(X)-1) * (X_centered).T @ (X_centered) 

            # 3. Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            eigenvectors = eigenvectors.T # Transpose eigenvectors

            # 4. Sorting eigenvalues by decreasing order and saving index
            eigenvalues_sorted_index = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)

            self.eigenvalues = eigenvalues[eigenvalues_sorted_index]
            self.components = eigenvectors[eigenvalues_sorted_index]

            # 5. Compute explained variance
            self.explained_variance_ratio = (self.eigenvalues / np.sum(self.eigenvalues)) * 100
            self.cumulative_explained_variance = np.cumsum(self.explained_variance)
        
        else:
            # Compute SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Store results
            self.components = Vt
            self.singular_values_ = S

            # Calculate explained variance
            self.explained_variance = ((S ** 2) / (X.shape[0] - 1)) * 100
            self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)

        # Keep only n_components (if specified)Selecting components
        if self.n_components != 0:
            self.components = self.components[0:self.n_components]

    def transform(self, X: Union[pd.DataFrame, np.ndarray, list], n_components: int = 0):
        """
        Transforms the input data to the specified number of components using the fitted PCA model.

        Parameters:
            X (Union[pd.DataFrame, np.ndarray, list]): The input data to be transformed.

        Returns:
            np.ndarray: The transformed data.

        """
        # Raise error if fit has not been called
        if self.mean is None or self.components is None:
            raise ValueError("Please fit the model first")
        
        # Transforms to specified number of components, if not specified, use all
        if n_components != 0:
            self.components = self.components[0:n_components]
        
        return (X - self.mean) @ self.components.T
    
    def fit_transform(self, 
                      X: Union[pd.DataFrame, np.ndarray, list], 
                      n_components: int = 0):
        """
        Applied pca to input, transforms to specified number of components and returns the reduced data.

        Parameters:
            X (Union[pd.DataFrame, np.ndarray, list]): The input data to fit the model on and transform.

        Returns:
            np.ndarray: The transformed data.

        """
        # Apply PCA and transform data
        self.n_components = n_components
        self.fit(X)
        return self.transform(X)
    