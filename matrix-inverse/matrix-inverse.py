import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, dtype=float)

    if A.shape[0] != A.shape[1] or A.ndim != 2:
        return None

    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
