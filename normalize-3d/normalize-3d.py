import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v)

    magnitude = np.linalg.norm(v, axis=-1, keepdims=True)

    if np.isscalar(magnitude):
        if magnitude == 0:
            return np.zeros_like(v)
        return v / magnitude
    else:
        safe_magnitude = np.where(magnitude == 0, 1, magnitude)
        normalized = v / safe_magnitude
        normalized[magnitude.squeeze() == 0] = 0
        return normalized

    