import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        X = np.asarray(matrix, dtype=float)
    except Exception:
        return None

    if X.ndim != 2:
        return None

    if axis not in (None, 0, 1):
        return None

    if not isinstance(norm_type, str):
        return None
    nt = norm_type.lower()
    if nt not in ("l2", "l1", "max"):
        return None

    absX = np.abs(X)

    if nt == "l2":
        norms = np.sqrt(np.sum(X * X, axis=axis, keepdims=True))
    elif nt == "l1":
        norms = np.sum(absX, axis=axis, keepdims=True)
    else:
        norms = np.max(absX, axis=axis, keepdims=True)

    norms_safe = np.where(norms == 0.0, 1.0, norms)
    return X / norms_safe
_matrix_normalization_ref = matrix_normalization
