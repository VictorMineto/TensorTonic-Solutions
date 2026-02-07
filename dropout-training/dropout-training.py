import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here

    x = np.asarray(x)
    if not (0.0 <= p < 1.0):
        raise ValueError("p must satisfy 0.0 <= p < 1.0")

    keep_prob = 1.0 - p


    r = rng.random(x.shape) if rng is not None else np.random.random(x.shape)

    keep = (r < keep_prob)

    scale = 1.0 / keep_prob
    dropout_pattern = keep.astype(x.dtype, copy=False) * scale

    out = x * dropout_pattern
    return out, dropout_pattern