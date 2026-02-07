def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here

    X = np.asarray(X)
    if X.ndim != 2:
        return None
    
    n, d = X.shape
    if not isinstance(k, (int, np.integer)) or k<1 or k>d:
        return None
    if n<2:
        return None
    
    Xc = X - X.mean(axis=0, keepdims=True)

    C = (Xc.T @ X) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]

    X_proj = Xc @ W
    return X_proj