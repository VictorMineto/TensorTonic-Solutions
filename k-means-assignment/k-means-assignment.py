import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here

    p = np.asarray(points, dtype=float)
    C = np.asarray(centroids, dtype=float)

    if p.ndim != 2 or C.ndim != 2 or p.shape[1] != C.shape[1] or C.shape[0] == 0:
        return None

    d2 = np.sum((p[:, None, :] - C[None, :, :]) ** 2, axis=2)

    return np.argmin(d2, axis=1).tolist()