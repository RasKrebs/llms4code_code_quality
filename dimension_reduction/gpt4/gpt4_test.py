import unittest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from long import PrincipalComponentAnalysis as Long_PCA
from medium import PrincipalComponentAnalysis as Medium_PCA
from small import pca_svd as small_pca_svd, pca_eigen as small_pca_eigen

# Test cases:
#   Can take Pandas DataFrame as input
#   Performs SVD equal to sklearn
#   Can reduce to a set number of components

np.random.seed(0)

class TestPCAs(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.X = np.random.rand(1000, 10)
        self.X_std = StandardScaler().fit_transform(self.X)
        self.ground_truth = PCA(n_components=5).fit_transform(self.X_std)
    
    def test_equal(self):
        long_pca = Long_PCA(n_components=5, decomposition_method='svd').fit_transform(self.X_std)
        medium_pca = Medium_PCA(n_components=5, decomposition_method='svd').fit_transform(self.X_std)
        small_pca = small_pca_svd(self.X_std, n_components=5)
        self.assertTrue(np.allclose(self.ground_truth, long_pca))
        self.assertTrue(np.allclose(self.ground_truth, medium_pca))
        self.assertTrue(np.allclose(self.ground_truth, small_pca))
        
    def takes_type(self):
        with self.assertRaises(TypeError):
            long_pca = long_pca(n_components=5).fit_transform(pd.DataFrame(self.X_std))
        with self.assertRaises(TypeError):
            medium_pca = medium_pca(n_components=5).fit_transform(pd.DataFrame(self.X_std))
        with self.assertRaises(TypeError):
            small_pca = small_pca_svd(pd.DataFrame(self.X_std), n_components=5)
        

if __name__ == '__main__':
    unittest.main()