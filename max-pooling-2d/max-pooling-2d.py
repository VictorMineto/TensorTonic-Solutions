def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    X = np.asarray(X)
    p = int(pool_size)
    if X.ndim != 2:
        raise ValueError("X doit être array 2D")
    if p <= 0:
        raise ValueError("p doit être entier positif")
    H, W = X.shape
    H_out = H // p
    W_out = W // p

    X_trim = X[:H_out * p, :W_out * p]
    out = X_trim.reshape(H_out, p, W_out, p).max(axis=(1, 3))

    return out