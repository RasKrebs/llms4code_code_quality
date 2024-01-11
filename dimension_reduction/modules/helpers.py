import numpy as np

def pca_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data
    X = np.random.rand(50000, 50)
    
    # Load BodyFat dataset
    # data = np.loadtxt('../data/bodyfat.txt', skiprows=11)
    
    return X