import numpy as np

def conv_forward_naive(x, w, b, pad, stride):
    # N: batch size
    # C: channels
    # H: height
    # W: width
    # _: same as C
    # F: filter number
    # HH: height of kernel
    # WW: width of kernel
    N, C, H, W   = x.shape
    F, _, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    out = np.zeros( (N, F, H_out, W_out) )

    # Pad the input
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    for n in range(N):
        for c in range(C):
            x_pad[n,c] = np.pad(x[n,c],(pad,pad),'constant', constant_values=(0,0))

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                # current matrix
                # (C, HH, WW)
                current_x_matrix = x_pad[n, :, i * stride: i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    # current filter matrix
                    # (C, HH, WW)
                    current_filter = w[f]
                    out[n,f,i,j] = np.sum(current_x_matrix*current_filter)

                out[n,:,i,j] = out[n,:,i,j] + b

    return out
