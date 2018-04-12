import numpy as np

def max_pool_forward_naive(x, pool_height, pool_width, stride):
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    out[n, c, h, w] = np.max(x[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width])

    return out

