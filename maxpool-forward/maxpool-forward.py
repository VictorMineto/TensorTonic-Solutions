import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X = np.asarray(X)
    H, W = X.shape

    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size

            window = X[h_start:h_end, w_start:w_end]

            output[i, j] = np.max(window)

    return output.tolist()
