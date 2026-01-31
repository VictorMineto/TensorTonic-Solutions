import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D array of shape matching X")

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    eps = 1e-12

    for _ in range(int(steps)):

        z = X @ w + b
        p = _sigmoid(z)

        err = p - y

        grad_w = X.T @ err / n_samples
        grad_b = np.sum(err) / n_samples

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b    