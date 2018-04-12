
def max_pool_backward_naive(x, dout, pool_height, pool_width, stride):
    N, C, H_out, W_out = dout.shape
    dx = np.zeros(x.shape)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    current_matrix = x[n, c, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width]
                    max_idx = np.unravel_index(np.argmax(current_matrix),current_matrix.shape)
                    dx[n, c, h * stride + max_idx[0], w * stride + max_idx[1]] += dout[n, c, h, w]


    return dx

