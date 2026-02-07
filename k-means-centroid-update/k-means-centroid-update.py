
import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    p = np.asarray(points, dtype=float)
    a = np.asarray(assignments)

    if p.ndim !=2:
        return None

    n, d = p.shape

    if a.ndim != 1 or a.shape[0] != n:
        return None
    if not isinstance(k, (int, np.integer)) or k <= 0:
        return None
    if n == 0:
        return None
    if np.any(a < 0) or np.any(a >= k):
        return None
    
    centroids_sum = np.zeros((k, d), dtype=float)
    np.add.at(centroids_sum, a, p)

    counts = np.bincount(a, minlength=k).astype(float)

    centroids = centroids_sum / np.where(counts[:, None] == 0.0, 1.0, counts[:, None])

    return centroids.tolist()