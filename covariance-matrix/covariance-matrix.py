import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return None
    
    n, d = X.shape

    if  n<2:
        return None
    
    mean = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (n - 1)

    return cov
